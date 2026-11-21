<template>
  <WizardHumanOverlay
    v-if="avatarInitialized"
    :command="design.avatar"
    :assets="assets"
    :playback="playback"
    :amplitude="amplitude"
    @status="avatarStatus = $event"
    @capabilities="rigCapabilities = $event"
  />
  <details class="presentation-panel">
    <summary>
      Self-design &amp; identity ocular evolution
      <span v-if="design?.personality?.identity_generation" class="gen-pill">
        Gen {{ design.personality.identity_generation }}
      </span>
    </summary>
    <p>{{ state }} · {{ avatarStatus }}</p>
    <p v-if="design?.personality?.design_intent" class="intent-line">
      <strong>Identity Intent:</strong> {{ design.personality.design_intent }}
    </p>
    <div v-if="design?.eye" class="eye-param-chips">
      <span class="param-chip">hue {{ Math.round(Number(design.eye.hue ?? 285)) }}°</span>
      <span class="param-chip">2nd {{ Math.round(Number(design.eye.secondary_hue ?? 192)) }}°</span>
      <span class="param-chip">hetero {{ Number(design.eye.heterochromia ?? 0.22).toFixed(2) }}</span>
      <span class="param-chip">pupil {{ Number(design.eye.pupil_aspect ?? 1.0).toFixed(2) }}</span>
      <span class="param-chip">striae {{ Number(design.eye.striae_detail ?? 0.62).toFixed(2) }}</span>
      <span class="param-chip">arms {{ Number(design.eye.spiral_arms ?? 2.0).toFixed(1) }}</span>
    </div>
    <p>
      W1z4rD evolves its eye appearance on the fly from its <code>ocular_identity</code> neuron pool,
      identity specification, and consolidated experiences across all twelve prerogatives.
    </p>
    <p v-if="!Object.keys(assets).length">No reviewed human rig is installed. The human overlay stays transparent.</p>
    <label>
      Experience / Design cue
      <textarea
        v-model="feedback"
        maxlength="2000"
        placeholder="Describe an experience, identity trait, or observed outcome to guide W1z4rD's ocular evolution…"
      />
    </label>
    <label>
      <input type="checkbox" v-model="learn" />
      Learn this reviewed design outcome (requires foundation admission)
    </label>
    <label v-if="learn">
      <input type="checkbox" v-model="reviewed" />
      I verified the observed outcome and checked all twelve prerogatives together.
    </label>
    <div class="action-row">
      <button @click="triggerAutonomousRedesign" :disabled="saving || evolving">
        {{ evolving ? 'Evolving Eye…' : '✨ Evolve Eye from Identity & Experience' }}
      </button>
      <button @click="submit" :disabled="saving || evolving || !feedback.trim() || (learn && !reviewed)">
        Send design feedback
      </button>
    </div>
    <p role="status">{{ result }}</p>
  </details>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount } from 'vue'
import WizardHumanOverlay from './WizardHumanOverlay.vue'

const props = defineProps<{
  brainId: string
  response?: Record<string, any>
  telemetry?: Record<string, unknown>
  playback?: { at_ms: number; present: boolean }
  amplitude: number
}>()

const emit = defineEmits<{ design: [value: Record<string, any>] }>()

const defaults = () => ({
  version: 1,
  eye: {
    hue: 285,
    secondary_hue: 192,
    heterochromia: 0.22,
    openness: 0.68,
    pupil_aspect: 1.0,
    pupil_scale: 1.0,
    iris_radius: 0.56,
    limbal_ring: 0.58,
    striae_detail: 0.62,
    collarette_glow: 0.45,
    sclera_tint: 0.14,
    cornea_gloss: 0.72,
    spiral_tightness: 0.16,
    spiral_arms: 2.0,
    nebula_strength: 0.8,
    particle_speed: 0.25,
  },
  avatar: {
    visible: false,
    asset_id: '',
    opacity: 0.55,
    x: 0.75,
    scale: 1,
    animation: '',
    morphs: {},
    visemes: [],
  },
  personality: {
    self_description: '',
    design_intent: '',
    identity_generation: 1,
    prerogative_alignment: {},
  },
})

const design = ref<Record<string, any>>(defaults())
const assets = ref<Record<string, any>>({})
const state = ref('Loading self-design interface')
const avatarStatus = ref('Human overlay transparent')
const feedback = ref('')
const learn = ref(false)
const reviewed = ref(false)
const saving = ref(false)
const evolving = ref(false)
const result = ref('')
const rigCapabilities = ref<Record<string, unknown>>({})
const avatarInitialized = ref(false)

watch(
  () => design.value.avatar?.visible,
  visible => {
    if (visible) avatarInitialized.value = true
  }
)

let controller: AbortController | undefined
let lastReport = 0
let alive = true

const csrf = () => decodeURIComponent(document.cookie.match(/(?:^|; )csrftoken=([^;]*)/)?.[1] || '')

async function load() {
  controller?.abort()
  const own = (controller = new AbortController())
  avatarInitialized.value = false
  design.value = defaults()
  assets.value = {}
  emit('design', design.value)
  try {
    const response = await fetch('/api/wizard-chat/self-design/', { signal: own.signal })
    const data = await response.json()
    if (!response.ok || !data.design) throw Error(data.error || 'Self-design interface unavailable')
    if (own.signal.aborted) return
    design.value = data.design
    assets.value = data.assets || {}
    state.value = data.gate?.ready
      ? 'Model visual control admitted'
      : 'Identity & neural-pool eye evolution active'
    emit('design', design.value)
  } catch (error) {
    if (!own.signal.aborted) state.value = String(error)
  }
}

watch(
  () => props.brainId,
  () => void load()
)

watch(
  () => props.response,
  response => {
    if (response?.applied && response.design) {
      design.value = response.design
      assets.value = response.assets || assets.value
      state.value =
        response.source === 'ocular_identity_pool'
          ? `Autonomous identity eye evolution applied (${response.reason || 'neural pool'})`
          : 'Native model design applied · awaiting reviewed outcome'
      emit('design', design.value)
    } else if (response?.reason) {
      state.value = response.reason
    }
  }
)

async function send(body: Record<string, unknown>) {
  const response = await fetch('/api/wizard-chat/self-design/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf() },
    body: JSON.stringify(body),
  })
  const value = await response.json()
  if (!response.ok) throw Error(value.error || 'Self-design request failed')
  return value
}

watch(
  () => props.telemetry,
  telemetry => {
    if (!telemetry || Date.now() - lastReport < 15000) return
    lastReport = Date.now()
    void send({
      kind: 'visual-observation',
      design: design.value,
      observed: { ...telemetry, rig: rigCapabilities.value },
    }).catch(() => {
      /* Render observation failure never becomes an inferred successful learning outcome. */
    })
  }
)

async function triggerAutonomousRedesign() {
  evolving.value = true
  result.value = ''
  try {
    const value = await send({
      kind: 'autonomous-redesign',
      feedback: feedback.value.trim(),
    })
    if (!alive) return
    if (value.applied && value.design) {
      design.value = value.design
      emit('design', design.value)
      const gen = value.design?.personality?.identity_generation ?? '?'
      result.value = `Eye redesigned on the fly (Gen ${gen}): ${value.reason || 'identity & experience shift'}.`
    } else {
      result.value = `Identity eye design retained: ${value.reason || 'current ocular morphology matches neural state'}.`
    }
  } catch (error) {
    if (alive) result.value = String(error)
  } finally {
    if (alive) evolving.value = false
  }
}

async function submit() {
  saving.value = true
  result.value = ''
  try {
    const value = await send({
      kind: 'visual-feedback',
      design: design.value,
      feedback: feedback.value,
      observed: props.telemetry || {},
      learn: learn.value,
      reviewed_experience: learn.value
        ? {
            reviewer: 'authenticated operator',
            outcome_verified: reviewed.value,
            checked_prerogatives: Array.from({ length: 12 }, (_, i) => `P${String(i + 1).padStart(2, '0')}`),
            conflicts: [],
            observed_outcome: { design_feedback: feedback.value },
          }
        : undefined,
    })
    if (alive) {
      result.value = value.consolidated
        ? 'Design outcome consolidated; retention reassessment required.'
        : 'Feedback recorded for the next response; no training performed.'
    }
  } catch (error) {
    if (alive) result.value = String(error)
  } finally {
    if (alive) saving.value = false
  }
}

onMounted(() => void load())
onBeforeUnmount(() => {
  alive = false
  controller?.abort()
})
</script>

<style scoped>
.presentation-panel {
  flex-shrink: 0;
  padding: 0.5rem 1rem;
  font-size: 0.74rem;
  color: #d9c6e5;
  max-height: 38vh;
  overflow: auto;
  border-top: 1px solid #bda4cd30;
}
summary {
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.45rem;
}
.gen-pill {
  font-size: 0.65rem;
  padding: 0.08rem 0.42rem;
  border-radius: 999px;
  background: #2b1a3d;
  border: 1px solid #a87fd466;
  color: #e9d5ff;
}
.intent-line {
  color: #e9d5ff;
  margin: 0.35rem 0;
}
.eye-param-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem;
  margin: 0.35rem 0;
}
.param-chip {
  font-size: 0.66rem;
  padding: 0.1rem 0.4rem;
  border-radius: 6px;
  background: #171126;
  border: 1px solid #68537188;
  color: #d8b4fe;
}
.action-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 0.4rem;
}
label {
  display: block;
  margin: 0.5rem 0;
}
textarea {
  display: block;
  width: 95%;
  min-height: 3rem;
  color: #e9d9ef;
  background: #100c1a;
  border: 1px solid #685371;
  border-radius: 8px;
  padding: 0.5rem;
}
button {
  color: #f1d8ec;
  background: #21152a;
  border: 1px solid #9b739f;
  border-radius: 12px;
  padding: 0.4rem 0.7rem;
  cursor: pointer;
}
button:disabled {
  opacity: 0.4;
  cursor: default;
}
</style>
