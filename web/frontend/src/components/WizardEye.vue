<template>
  <div
    ref="host"
    class="eye"
    :class="`eye-state-${lifecycleState}`"
    @pointermove="look"
    @pointerleave="pointerGaze = [0, 0]"
  >
    <span v-if="error" class="render-error">{{ error }}</span>
    <span class="signal-label">{{ signalLabel }}</span>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import * as THREE from 'three'
import { vectorAt, type EyeSignal } from '@/utils/eyeSignal'
import type { EyeAssociation } from '@/utils/eyeAssociations'
import type { ObservationCue } from '@/utils/observation'

export interface NeuralSentiment {
  pool_grounded?: boolean
  dominant_sentiment?: string
  valence?: number
  arousal?: number
  warmth?: number
  focus?: number
  wonder?: number
  caution?: number
  fired_neurons?: Array<{ neuron: string; strength: number }>
}

const props = defineProps<{
  concepts: string[]
  messageId: string
  busy: boolean
  amplitude: number
  online?: boolean | null
  cognitiveState?: Record<string, any> | null
  sentiment?: NeuralSentiment | null
  signal?: EyeSignal
  playback?: { at_ms: number; present: boolean }
  association?: EyeAssociation
  observation?: ObservationCue
  design?: Record<string, number>
}>()

const emit = defineEmits<{ telemetry: [value: Record<string, unknown>] }>()
const host = ref<HTMLDivElement>()
const error = ref('')

type EyeLifecycleState = 'connecting' | 'offline' | 'recovering' | 'settling_sleep' | 'rem_sleep' | 'waking' | 'awake'

const lifecycleState = ref<EyeLifecycleState>('awake')
let wakeStartedPerfMs = -Infinity
let recoveryStartedPerfMs = -Infinity
let backendWakeStartedPerfMs = -Infinity
let lastBackendWakeSec: number | undefined
let acceptBackendWake = true
let localSleepStartedPerfMs = 0
let previousRest: 'sleep' | 'unavailable' | undefined
let lidClosing: { elapsed: number; from: number } | undefined

function computeSettlingElapsedSec(nowPerfMs: number): number {
  const cogDur = Number(props.cognitiveState?.mode_duration_seconds)
  const cogMode = String(props.cognitiveState?.mode || '')
  if ((cogMode === 'SLEEPING_SETTLING' || cogMode === 'CHECKPOINT_LOCKED') && Number.isFinite(cogDur) && cogDur >= 0) {
    return cogDur
  }
  if (localSleepStartedPerfMs > 0) {
    return Math.max(0, (nowPerfMs - localSleepStartedPerfMs) / 1000)
  }
  return 0
}

function updateLifecycleState(nowPerfMs: number): { state: EyeLifecycleState; settlingSec: number; wakeElapsedSec: number } {
  const cogMode = String(props.cognitiveState?.mode || 'IDLE_READY')
  const isConnecting = props.online === null
  const isOffline = props.online === false || cogMode === 'OFFLINE'
  const backendWakeSec = props.cognitiveState?.seconds_since_wake
  if (acceptBackendWake && typeof backendWakeSec === 'number' && Number.isFinite(backendWakeSec) && backendWakeSec >= 0
      && backendWakeSec !== lastBackendWakeSec) {
    lastBackendWakeSec = backendWakeSec
    backendWakeStartedPerfMs = nowPerfMs - backendWakeSec * 1000
  }

  if (isConnecting || isOffline) {
    previousRest = 'unavailable'
    localSleepStartedPerfMs = 0
    lifecycleState.value = isConnecting ? 'connecting' : 'offline'
    return { state: lifecycleState.value, settlingSec: 0, wakeElapsedSec: 999 }
  }

  if (cogMode === 'SLEEPING_SETTLING' || cogMode === 'CHECKPOINT_LOCKED') {
    if (!localSleepStartedPerfMs) localSleepStartedPerfMs = nowPerfMs
    previousRest = 'sleep'
    recoveryStartedPerfMs = -Infinity
    const settlingSec = computeSettlingElapsedSec(nowPerfMs)
    const isRem = Boolean(props.cognitiveState?.is_rem_sleep) || settlingSec >= 3600
    lifecycleState.value = isRem ? 'rem_sleep' : 'settling_sleep'
    return { state: lifecycleState.value, settlingSec, wakeElapsedSec: 999 }
  }

  if (previousRest) {
    if (previousRest === 'sleep') {
      wakeStartedPerfMs = nowPerfMs
    } else {
      recoveryStartedPerfMs = nowPerfMs
      wakeStartedPerfMs = backendWakeStartedPerfMs = -Infinity
      // A sampled sleep age carried across reconnect must not restart a sleepy wake.
      acceptBackendWake = false
    }
    previousRest = undefined
    localSleepStartedPerfMs = 0
  }
  if (nowPerfMs - recoveryStartedPerfMs < 1450) {
    lifecycleState.value = 'recovering'
    return { state: 'recovering', settlingSec: 0, wakeElapsedSec: 999 }
  }
  const wakeElapsedSec = (nowPerfMs - Math.max(wakeStartedPerfMs, backendWakeStartedPerfMs)) / 1000

  if (wakeElapsedSec < 5.6) {
    lifecycleState.value = 'waking'
    return { state: 'waking', settlingSec: 0, wakeElapsedSec }
  }

  lifecycleState.value = 'awake'
  return { state: 'awake', settlingSec: 0, wakeElapsedSec }
}

const signalLabel = computed(() => {
  if (lifecycleState.value === 'connecting') return 'Connecting · Searching'
  if (lifecycleState.value === 'offline') return 'Node offline · Standby'
  if (lifecycleState.value === 'recovering') return 'Connection restored'
  if (lifecycleState.value === 'rem_sleep') return 'Deep consolidation (1h+) · REM sleep'
  if (lifecycleState.value === 'settling_sleep') return 'Settling synapses · Sleeping'
  if (lifecycleState.value === 'waking') return 'Waking up…'
  if (props.sentiment?.pool_grounded && props.sentiment.dominant_sentiment) {
    return `Neural affect · ${props.sentiment.dominant_sentiment}`
  }
  return props.signal?.source === 'model-vector'
    ? 'Model vector trajectory'
    : props.association
    ? `Recalled silhouette · ${props.association.concept}`
    : props.concepts.length
    ? 'Concept signature · snapshot'
    : 'Ambient visual · awaiting concepts'
})

let pointerGaze: [number, number] = [0, 0]
let renderer: THREE.WebGLRenderer | undefined
let scene: THREE.Scene
let camera: THREE.PerspectiveCamera
let resize: ResizeObserver
let frame = 0
let start = performance.now()
let last = 0
let frames = 0
let reportAt = 0

let eye: THREE.Mesh
let cornea: THREE.Mesh
let stars: THREE.Points
let nebula: THREE.Mesh
let positions: Float32Array
let homes: Float32Array
let velocities: Float32Array
let pupilTexture: THREE.CanvasTexture | undefined

const preview = document.createElement('canvas')
preview.width = preview.height = 64
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)')

const uniforms = {
  time: { value: 0 },
  voice: { value: 0 },
  hue: { value: 285 / 360 },
  secondaryHue: { value: 198 / 360 },
  heterochromia: { value: 0.28 },
  openness: { value: 0.65 },
  squeeze: { value: 0.0 },
  sleepMode: { value: 0.0 }, // 0=awake or standby, 2=NREM settling, 3=REM
  standby: { value: 0.0 },
  pupilAspect: { value: 1.0 },
  pupilScale: { value: 1.0 },
  irisRadius: { value: 0.58 },
  limbalRing: { value: 0.75 },
  striaeDetail: { value: 1.0 },
  collaretteGlow: { value: 0.42 },
  scleraTint: { value: 0.08 },
  corneaGloss: { value: 0.88 },
  tightness: { value: 0.16 },
  spiralArms: { value: 1.0 },
  nebula: { value: 0.8 },
  valence: { value: 0.0 },
  arousal: { value: 0.25 },
  warmth: { value: 0.4 },
  focus: { value: 0.5 },
  caution: { value: 0.0 },
  gaze: { value: new THREE.Vector2() },
  vector: { value: new THREE.Vector4() },
  pupil: { value: null as THREE.Texture | null },
  hasPupil: { value: 0 },
}

const vertex = `
varying vec3 localPosition;
varying vec3 normalView;
varying vec3 viewPosition;
varying vec2 vScreen;
void main() {
  localPosition = position;
  normalView = normalize(normalMatrix * normal);
  vec4 p = modelViewMatrix * vec4(position, 1.0);
  viewPosition = -p.xyz;
  gl_Position = projectionMatrix * p;
  vScreen = gl_Position.xy / gl_Position.w;
}
`

const noise = `
float hash(vec3 p) {
  return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}
float hash2(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float noise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash(i), hash(i + vec3(1.0, 0.0, 0.0)), f.x),
        mix(hash(i + vec3(0.0, 1.0, 0.0)), hash(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),
    mix(mix(hash(i + vec3(0.0, 0.0, 1.0)), hash(i + vec3(1.0, 0.0, 1.0)), f.x),
        mix(hash(i + vec3(0.0, 1.0, 1.0)), hash(i + vec3(1.0, 1.0, 1.0)), f.x), f.y),
    f.z
  );
}
float fbm(vec3 p) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 5; i++) {
    s += a * noise(p);
    p = p * 2.04 + 17.1;
    a *= 0.5;
  }
  return s;
}
vec3 hueColor(float h) {
  vec3 p = abs(fract(vec3(h) + vec3(0.0, 0.666667, 0.333333)) * 6.0 - 3.0);
  return mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), 0.78);
}
`

const irisFragment = `
precision highp float;
varying vec3 localPosition;
varying vec3 normalView;
varying vec3 viewPosition;

uniform float time, voice, hue, secondaryHue, heterochromia, openness, squeeze, sleepMode, standby;
uniform float pupilAspect, pupilScale, irisRadius, limbalRing, striaeDetail, collaretteGlow;
uniform float scleraTint, corneaGloss, tightness, spiralArms, hasPupil;
uniform float valence, arousal, warmth, focus, caution;
uniform vec2 gaze;
uniform vec4 vector;
uniform sampler2D pupil;

${noise}

void main() {
  vec2 p = localPosition.xy;

  // Outer anatomical ocular socket mask
  float socketR = length(vec2(p.x * 0.90, p.y * 1.06));
  if (socketR > 0.99) discard;
  float socketFeather = smoothstep(0.99, 0.92, socketR);

  // Natural almond eyelid curve + affective expression shaping
  float lidArch = pow(max(0.0, 1.0 - p.x * p.x * 1.04), 0.72);
  // Closed-seam resting curve: slight serene downward arc, pulled tighter during sleep squeezes
  float closedSeamY = (-0.035 + 0.025 * squeeze) * lidArch + 0.015 * valence * lidArch;

  // The aperture can reach a closed seam during sleep or a blink.
  float openAmt = clamp(openness, 0.0, 1.15);
  // Warmth/positive valence lifts the lower lid (Duchenne smile); caution narrows slightly
  float smileLift = max(0.0, valence) * 0.075 + warmth * 0.035;
  float cautionNarrow = caution * 0.06;

  float upperSpan = max(0.0, 0.78 * openAmt - cautionNarrow * 0.4) * lidArch;
  float lowerSpan = max(0.0, 0.50 * openAmt - smileLift - cautionNarrow * 0.5) * lidArch;

  float upperLidY = closedSeamY + upperSpan;
  float lowerLidY = closedSeamY - lowerSpan;

  float edgeDist = min(upperLidY - p.y, p.y - lowerLidY);
  vec2 eyeCenter = vec2(gaze.x * 0.24, gaze.y * 0.15);

  // ── 3D Anatomical Eyelids (visible whenever partially lidded or completely shut) ──
  if (openAmt <= 0.0015 || edgeDist <= 0.0) {
    vec3 n = normalize(normalView);
    vec3 lightDir = normalize(vec3(-0.45, 0.72, 1.65));

    // Subsurface eyeball dome pressing gently beneath the closed eyelid (reveals NREM drift & REM bursts!)
    float subGlobeDist = length((p - eyeCenter) * vec2(1.0, 1.12));
    float subGlobeDome = exp(-subGlobeDist * subGlobeDist * 4.2);

    // Closed palpebral fissure seam & eyelash shadow line
    float seamDist = abs(p.y - closedSeamY);
    float seamLine = exp(-seamDist * (95.0 + 55.0 * squeeze)) * lidArch;

    // Supratarsal upper-lid fold crease + squeeze micro-creases ("squeezes here and there")
    float foldTargetY = upperLidY + (0.14 - 0.045 * squeeze) * lidArch + 0.03;
    float upperFold = exp(-pow((p.y - foldTargetY) * 26.0, 2.0)) * smoothstep(0.05, 0.55, lidArch);
    float squeezeCrease1 = squeeze * exp(-pow((p.y - (closedSeamY + 0.055 * lidArch)) * 42.0, 2.0)) * lidArch;
    float squeezeCrease2 = squeeze * exp(-pow((p.y - (closedSeamY - 0.048 * lidArch)) * 44.0, 2.0)) * lidArch;
    float canthalWrinkle = squeeze * smoothstep(0.45, 0.88, abs(p.x)) * exp(-abs(p.y - closedSeamY) * 18.0)
                         * (0.5 + 0.5 * sin((p.y - closedSeamY) * 95.0));

    vec3 baseTone = hueColor(hue);
    vec3 lidSkinDark = vec3(0.045, 0.036, 0.072);
    vec3 lidSkinMid  = mix(vec3(0.11, 0.088, 0.165), baseTone * 0.24, 0.35);

    // REM keeps its subsurface shimmer while standby has a softer light.
    float liveFactor = mix(1.0, 0.72, standby);
    float diff = (0.32 + 0.68 * max(0.0, dot(n, lightDir))) * liveFactor;
    vec3 lidColor = mix(lidSkinDark, lidSkinMid, diff * (0.62 + 0.38 * subGlobeDome));

    // Subsurface globe highlight shifting softly during NREM / rapidly during REM
    lidColor += mix(vec3(0.10, 0.09, 0.16), baseTone * 0.22, 0.5) * subGlobeDome * 0.35 * liveFactor;
    if (sleepMode > 2.5) {
      // REM sleep: subtle dream-activity shimmer beneath shut lid (never opening!)
      float remPulse = 0.5 + 0.5 * sin(time * 5.2 + subGlobeDist * 8.0);
      lidColor += hueColor(secondaryHue) * subGlobeDome * remPulse * 0.09;
    }

    // Darken anatomical folds, squeeze creases, and closed lash seam
    lidColor *= 1.0 - 0.45 * upperFold;
    lidColor *= 1.0 - 0.42 * (squeezeCrease1 + squeezeCrease2 + canthalWrinkle * 0.6);
    lidColor = mix(lidColor, vec3(0.012, 0.008, 0.022), clamp(seamLine * (0.88 + 0.12 * squeeze), 0.0, 0.96));

    gl_FragColor = vec4(pow(max(lidColor, vec3(0.0)), vec3(0.78)), socketFeather);
    return;
  }

  // ── Open / Waking Eye Interior (Iris, Heterochromia, Adaptive Pupil, Sclera) ──
  vec2 irisCoord = p - eyeCenter;
  irisCoord.y *= 1.02;

  float r = length(irisCoord);
  float a = atan(irisCoord.y, irisCoord.x);

  // 1. Multi-frequency radial stromal fibers & Crypts of Fuchs (scaled by striaeDetail)
  float fiberNoise = fbm(vec3(cos(a * 2.0) * 16.0, sin(a * 2.0) * 16.0, r * 24.0 * striaeDetail));
  float fineStriae = sin(a * (165.0 + 45.0 * striaeDetail) + noise(vec3(a * 14.0, r * 28.0, 2.5)) * 6.0 + r * 30.0);

  // 2. Pupillary/ciliary collarette + central heterochromia
  float collaretteR = irisRadius * 0.48 + 0.028 * sin(a * 7.0 + fiberNoise * 3.0);
  float collaretteRing = smoothstep(0.042, 0.0, abs(r - collaretteR));

  float cryptMask = smoothstep(0.62, 0.95, fbm(vec3(cos(a * 5.0) * 8.0, sin(a * 5.0) * 8.0, r * 14.0)));

  // Affective warmth subtly enriches inner iris golden-rose glow
  vec3 outerTone = hueColor(hue + valence * 0.018);
  vec3 innerTone = mix(outerTone, hueColor(secondaryHue), clamp(heterochromia, 0.0, 1.0));
  vec3 zoneTone  = mix(innerTone, outerTone, smoothstep(collaretteR * 0.72, irisRadius * 0.88, r));

  vec3 darkPigment = mix(vec3(0.035, 0.022, 0.048), zoneTone * 0.24, fiberNoise);
  vec3 vibrantPigment = mix(zoneTone * 0.68, vec3(0.88, 0.82, 0.96), (fineStriae * 0.35 + 0.5) * clamp(striaeDetail * 0.85, 0.4, 1.35));
  vec3 irisColor = mix(darkPigment, vibrantPigment, smoothstep(0.15, irisRadius * 0.78, r));

  vec3 glowTint = mix(hueColor(secondaryHue), vec3(1.0, 0.82, 0.58), warmth * 0.45);
  irisColor += glowTint * collaretteRing * (0.35 + 0.55 * collaretteGlow);
  irisColor = mix(irisColor, darkPigment * 0.55, cryptMask * 0.48 * clamp(striaeDetail, 0.5, 1.5));

  // 3. Limbal Ring
  float limbusInner = irisRadius - 0.07;
  float limbusOuter = irisRadius + 0.02;
  float limbus = smoothstep(limbusOuter, limbusInner, r);
  float limbusDarkRim = smoothstep(limbusInner, irisRadius, r) * (1.0 - smoothstep(irisRadius, limbusOuter, r));
  irisColor = mix(irisColor * (1.0 - 0.72 * limbalRing), irisColor, 1.0 - limbusDarkRim * clamp(limbalRing, 0.15, 1.0));

  // 4. Multi-arm logarithmic vector spiral
  float deformation = 0.04 * (vector.x * sin(a * 2.0) + vector.y * cos(a * 3.0));
  float phase = a * max(1.0, floor(spiralArms + 0.5)) - log(max(0.025, r) / 0.025) / max(0.05, tightness) - time * 0.03 + deformation;
  float angularDist = abs(atan(sin(phase), cos(phase)));
  float spiral = exp(-pow(angularDist * max(0.07, r) * 95.0, 2.0))
               * smoothstep(0.11, 0.17, r)
               * (1.0 - smoothstep(irisRadius - 0.07, irisRadius - 0.01, r));
  vec3 spiralColor = mix(hueColor(secondaryHue), vec3(0.72, 0.86, 1.0), 0.5 + 0.5 * sin(log(max(0.02, r)) * 4.0));
  irisColor += spiralColor * spiral * (1.25 + 0.4 * collaretteGlow);

  // 5. Sclera (porcelain to celestial obsidian via scleraTint)
  vec3 porcelainSclera = vec3(0.92, 0.91, 0.95);
  vec3 celestialSclera = mix(vec3(0.06, 0.08, 0.15), outerTone * 0.18, 0.4);
  vec3 scleraBase = mix(porcelainSclera, celestialSclera, clamp(scleraTint, 0.0, 0.85));
  float scleraShadow = smoothstep(0.0, 0.34, edgeDist);
  float scleraTexture = fbm(vec3(p * 8.0, 1.0));
  vec3 sclera = mix(scleraBase * 0.76, scleraBase, scleraShadow * 0.62 + scleraTexture * 0.38);

  float capNoise = fbm(vec3(p * 22.0, 4.0));
  float vessel = pow(max(0.0, sin(a * 18.0 + capNoise * 9.0 + r * 22.0)), 28.0) * smoothstep(irisRadius * 0.9, 0.95, r);
  sclera = mix(sclera, mix(vec3(0.72, 0.22, 0.28), hueColor(secondaryHue) * 0.6, scleraTint), vessel * (0.42 + 0.2 * caution));

  vec3 eyeColor = mix(sclera, irisColor, limbus);

  // 6. Adaptive Pupil Dynamics (circular-to-slit aspect + neural arousal/focus/voice modulation)
  float affectivePupil = (arousal - 0.35) * 0.032 - (focus - 0.5) * 0.018;
  float pupilRadius = clamp((0.132 * pupilScale) + voice * 0.024 + affectivePupil + 0.006 * sin(time * 0.45), 0.065, irisRadius * 0.68);
  vec2 pupilCoord = vec2(irisCoord.x / max(0.28, pupilAspect), irisCoord.y);
  float pr = length(pupilCoord);
  float pupilFeather = 0.01;
  float pupilMask = 1.0 - smoothstep(pupilRadius - pupilFeather, pupilRadius + pupilFeather, pr);

  if (hasPupil > 0.5) {
    vec2 uv = irisCoord / 0.48 + 0.5;
    pupilMask = texture2D(pupil, uv).a * step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
  }

  eyeColor = mix(eyeColor, vec3(0.005, 0.003, 0.009), pupilMask);

  // 7. Lighting & Wet Cornea Specular Reflections
  vec3 n = normalize(normalView);
  vec3 v = normalize(viewPosition);
  vec3 light1 = normalize(vec3(-0.55, 0.75, 1.8));
  vec3 light2 = normalize(vec3(0.65, -0.4, 1.2));

  float diff = 0.35 + 0.65 * max(0.0, dot(n, light1));
  eyeColor *= diff;

  vec3 h1 = normalize(light1 + v);
  float spec1 = pow(max(0.0, dot(n, h1)), 120.0);
  vec3 h2 = normalize(light2 + v);
  float spec2 = pow(max(0.0, dot(n, h2)), 45.0);
  eyeColor += vec3(0.95, 0.98, 1.0) * spec1 * 0.85 * corneaGloss;
  eyeColor += vec3(0.6, 0.7, 0.9) * spec2 * 0.25 * corneaGloss;

  // Upper/lower eyelid margin shadow & wet tear-film meniscus
  eyeColor *= smoothstep(0.0, 0.04, edgeDist);
  eyeColor += vec3(0.42, 0.28, 0.36) * exp(-abs(edgeDist - 0.013) * 110.0) * 0.32;

  gl_FragColor = vec4(pow(max(eyeColor, vec3(0.0)), vec3(0.72)), socketFeather);
}
`

function look(event: PointerEvent) {
  if (!host.value) return
  const rect = host.value.getBoundingClientRect()
  pointerGaze = [
    (event.clientX - rect.left) / rect.width - 0.5,
    (event.clientY - rect.top) / rect.height - 0.5,
  ]
}

function association() {
  pupilTexture?.dispose()
  pupilTexture = undefined
  uniforms.hasPupil.value = 0
  const shape = props.association?.silhouette
  if (!shape) return
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 256
  const ctx = canvas.getContext('2d')!
  const scale = 230 / Math.max(shape.width, shape.height)
  ctx.translate(128, 128)
  ctx.scale(scale, scale)
  ctx.translate(-shape.width / 2, -shape.height / 2)
  ctx.fillStyle = 'white'
  ctx.fill(new Path2D(shape.path), 'evenodd')
  pupilTexture = new THREE.CanvasTexture(canvas)
  uniforms.pupil.value = pupilTexture
  uniforms.hasPupil.value = 1
}

watch(() => props.association, association)
watch(() => props.messageId, () => { start = performance.now() })

function circularHueLerp(curNorm: number, targetNorm: number, alpha: number): number {
  const diff = ((targetNorm - curNorm + 1.5) % 1.0) - 0.5
  return (curNorm + diff * alpha + 1.0) % 1.0
}

/** Periodic organic eyelid squeezes during NREM/REM sleep ("squeezes here and there, never open"). */
function computeSleepSqueeze(t: number, isRem: boolean): number {
  const w1 = Math.pow(Math.max(0, Math.sin(t * 0.37 + 1.1)), 18)
  const w2 = Math.pow(Math.max(0, Math.sin(t * 0.23 + 4.2)), 24)
  const remTwitch = isRem ? 0.35 * Math.pow(Math.max(0, Math.sin(t * 1.9 + 0.5)), 16) : 0
  return Math.min(1.0, w1 * 0.85 + w2 * 0.7 + remTwitch)
}

/** Groggy multi-step sleepy eye waking curve over 5.6 seconds. */
function computeWakingCurve(wakeSec: number, steadyOpen: number): { open: number; squeeze: number; droopY: number; pupilBoost: number } {
  if (wakeSec < 0.55) {
    // Initial pre-wake sleepy squeeze while still shut
    const p = wakeSec / 0.55
    return { open: 0.0, squeeze: Math.sin(p * Math.PI) * 0.55, droopY: -0.10, pupilBoost: 1.22 }
  }
  if (wakeSec < 1.35) {
    // Crack eyelids open sleepily to ~0.18
    const p = (wakeSec - 0.55) / 0.80
    return { open: 0.18 * Math.sin(p * Math.PI * 0.5), squeeze: 0.08 * (1 - p), droopY: -0.09, pupilBoost: 1.18 }
  }
  if (wakeSec < 1.95) {
    // Heavy sleepy blink back shut
    const p = (wakeSec - 1.35) / 0.60
    const dip = Math.cos(p * Math.PI * 2) * 0.5 + 0.5
    return { open: 0.02 + 0.16 * dip, squeeze: (1 - dip) * 0.25, droopY: -0.07, pupilBoost: 1.14 }
  }
  if (wakeSec < 3.45) {
    // Half-lidded sleepy state slowly rising
    const p = (wakeSec - 1.95) / 1.50
    return { open: 0.18 + 0.26 * p, squeeze: 0.0, droopY: -0.06 * (1 - p), pupilBoost: 1.12 - 0.08 * p }
  }
  if (wakeSec < 3.95) {
    // Second gentle clearing blink
    const p = (wakeSec - 3.45) / 0.50
    const blink = Math.abs(p - 0.45) / 0.55
    return { open: Math.max(0.05, 0.48 * Math.min(1, blink)), squeeze: 0.06 * (1 - Math.min(1, blink)), droopY: -0.015, pupilBoost: 1.03 }
  }
  const p = Math.min(1, (wakeSec - 3.95) / 1.65)
  const ease = p * p * (3 - 2 * p)
  return { open: 0.48 + (steadyOpen - 0.48) * ease, squeeze: 0.0, droopY: 0.0, pupilBoost: 1.0 + 0.03 * (1 - ease) }
}

/** Spontaneous physiological waking blinks (160ms) modulated by neural arousal/focus. */
function computeSpontaneousBlink(t: number, arousal: number, focus: number): number {
  const rate = 0.78 + arousal * 0.45 - focus * 0.22
  const phase = (t * rate * 0.19) % 1.0
  // Sharp blink pulse around phase = 0.92
  if (phase > 0.90 && phase < 0.955) {
    const u = (phase - 0.90) / 0.055
    return Math.sin(u * Math.PI)
  }
  return 0.0
}

function closeLids(frameMs: number) {
  // Start at the rendered aperture, including a partially open lid mid-transition.
  if (lidClosing) {
    // A long render stall should not skip directly to the closed pose.
    lidClosing.elapsed += Math.min(frameMs, 125)
  } else {
    lidClosing = { elapsed: 0, from: uniforms.openness.value }
  }
  lidClosing.elapsed = reduced.matches ? 900 : Math.min(900, lidClosing.elapsed)
  const progress = lidClosing.elapsed / 900
  const eased = progress * progress * (3 - 2 * progress)
  uniforms.openness.value = lidClosing.from * (1 - eased)
}

function animate(time: number) {
  frame = requestAnimationFrame(animate)
  if (!renderer || time - last < (reduced.matches ? 200 : 33) || document.hidden) return
  const frameMs = time - last || 33
  const dt = Math.min(0.05, frameMs / 1000)
  last = time
  frames++

  const { state, wakeElapsedSec } = updateLifecycleState(time)
  const d = props.design || {}
  const sent = props.sentiment || {}

  // Smoothly morph all 16 identity self-design parameters on the fly
  uniforms.hue.value = circularHueLerp(uniforms.hue.value, ((d.hue ?? 285) % 360) / 360, 0.07)
  uniforms.secondaryHue.value = circularHueLerp(uniforms.secondaryHue.value, ((d.secondary_hue ?? 198) % 360) / 360, 0.07)
  uniforms.heterochromia.value += ((d.heterochromia ?? 0.28) - uniforms.heterochromia.value) * 0.07
  uniforms.pupilAspect.value += ((d.pupil_aspect ?? 1.0) - uniforms.pupilAspect.value) * 0.08
  uniforms.irisRadius.value += ((d.iris_radius ?? 0.58) - uniforms.irisRadius.value) * 0.07
  uniforms.limbalRing.value += ((d.limbal_ring ?? 0.75) - uniforms.limbalRing.value) * 0.07
  uniforms.striaeDetail.value += ((d.striae_detail ?? 1.0) - uniforms.striaeDetail.value) * 0.07
  uniforms.collaretteGlow.value += ((d.collarette_glow ?? 0.42) - uniforms.collaretteGlow.value) * 0.07
  uniforms.scleraTint.value += ((d.sclera_tint ?? 0.08) - uniforms.scleraTint.value) * 0.07
  uniforms.corneaGloss.value += ((d.cornea_gloss ?? 0.88) - uniforms.corneaGloss.value) * 0.07
  uniforms.tightness.value += ((d.spiral_tightness ?? 0.16) - uniforms.tightness.value) * 0.07
  uniforms.spiralArms.value += ((d.spiral_arms ?? 1.0) - uniforms.spiralArms.value) * 0.08
  uniforms.nebula.value += ((d.nebula_strength ?? 0.8) - uniforms.nebula.value) * 0.06

  // Smoothly interpolate emergent neural-pool sentiment coordinates
  const sValence = Number.isFinite(sent.valence) ? Number(sent.valence) : 0.0
  const sArousal = Number.isFinite(sent.arousal) ? Number(sent.arousal) : 0.25
  const sWarmth  = Number.isFinite(sent.warmth)  ? Number(sent.warmth)  : 0.40
  const sFocus   = Number.isFinite(sent.focus)   ? Number(sent.focus)   : 0.50
  const sWonder  = Number.isFinite(sent.wonder)  ? Number(sent.wonder)  : 0.20
  const sCaution = Number.isFinite(sent.caution) ? Number(sent.caution) : 0.0

  uniforms.valence.value += (sValence - uniforms.valence.value) * 0.08
  uniforms.arousal.value += (sArousal - uniforms.arousal.value) * 0.08
  uniforms.warmth.value  += (sWarmth  - uniforms.warmth.value)  * 0.08
  uniforms.focus.value   += (sFocus   - uniforms.focus.value)   * 0.08
  uniforms.caution.value += (sCaution - uniforms.caution.value) * 0.08

  const t = reduced.matches ? 0 : time / 1000
  const unavailable = state === 'offline' || state === 'connecting'
  const ease = reduced.matches ? 1 : 1 - Math.exp(-dt * 6)
  uniforms.time.value = t
  uniforms.standby.value += ((unavailable ? 1 : 0) - uniforms.standby.value) * ease
  if (state !== 'offline' && state !== 'settling_sleep' && state !== 'rem_sleep') lidClosing = undefined

  if (unavailable) {
    const connecting = state === 'connecting'
    const phase = t * (connecting ? 1.45 : 0.72)
    uniforms.sleepMode.value = 0.0
    if (connecting) {
      const open = 0.43 + (reduced.matches ? 0 : 0.045 * Math.sin(phase))
      uniforms.openness.value += (open - uniforms.openness.value) * ease
    } else {
      closeLids(frameMs)
    }
    const squeeze = !connecting && !reduced.matches ? 0.06 * (1 + Math.sin(phase)) : 0
    uniforms.squeeze.value += (squeeze - uniforms.squeeze.value) * ease
    uniforms.voice.value *= 1 - ease
    uniforms.pupilScale.value += (0.86 - uniforms.pupilScale.value) * ease
    const x = reduced.matches ? 0 : Math.sin(phase) * (connecting ? 0.42 : 0.22)
    const y = reduced.matches ? 0 : -0.1 + Math.sin(phase * 0.54) * 0.07
    const gaze = uniforms.gaze.value
    gaze.set(gaze.x + (x - gaze.x) * ease, gaze.y + (y - gaze.y) * ease)
    eye.rotation.set(-gaze.y * 0.06, gaze.x * 0.09, 0)
    cornea.rotation.copy(eye.rotation)
  } else if (state === 'settling_sleep' || state === 'rem_sleep') {
    // Settle into a closed lid, then keep any sleep activity beneath it.
    const isRem = state === 'rem_sleep'
    uniforms.sleepMode.value = isRem ? 3.0 : 2.0
    closeLids(frameMs)
    uniforms.voice.value *= 1 - ease

    const squeezeTarget = computeSleepSqueeze(t, isRem)
    uniforms.squeeze.value += (squeezeTarget - uniforms.squeeze.value) * 0.16

    let sleepGazeX = 0
    let sleepGazeY = 0
    if (isRem) {
      // Phasic REM bursts (rapid darting eye movements under the shut eyelid) + brief tonic pauses
      const burstGate = Math.max(0, Math.sin(t * 0.62) + 0.25 * Math.sin(t * 1.43))
      if (burstGate > 0.18) {
        sleepGazeX = 0.36 * Math.sin(t * 11.4) + 0.18 * Math.cos(t * 19.7)
        sleepGazeY = 0.20 * Math.sin(t * 8.9 + 0.7) + 0.10 * Math.sin(t * 16.3)
      } else {
        sleepGazeX = 0.08 * Math.sin(t * 0.9)
        sleepGazeY = -0.04 + 0.05 * Math.cos(t * 0.7)
      }
    } else {
      // Peaceful NREM settling: slow gentle drift beneath closed lid
      sleepGazeX = 0.13 * Math.sin(t * 0.55) + 0.04 * Math.sin(t * 1.3)
      sleepGazeY = -0.03 + 0.07 * Math.cos(t * 0.42)
    }

    uniforms.gaze.value.lerp(new THREE.Vector2(sleepGazeX, sleepGazeY), isRem ? 0.32 : 0.08)
    eye.rotation.y = uniforms.gaze.value.x * (isRem ? 0.14 : 0.06)
    eye.rotation.x = -uniforms.gaze.value.y * (isRem ? 0.10 : 0.045)
    cornea.rotation.copy(eye.rotation)
  } else {
    // 4. WAKING (sleepy eye waking up) & 5. AWAKE (conversational sentiment + live gaze)
    uniforms.sleepMode.value = 0.0
    uniforms.voice.value = props.amplitude

    const expression = props.association?.expression
    const baseOpen = d.openness ?? 0.66
    const sentimentOpenDelta =
      sWonder * 0.16 +
      (sArousal - 0.35) * 0.09 -
      sCaution * 0.05 +
      (props.busy ? 0.10 : 0) +
      (expression === 'wonder' ? 0.16 : expression === 'amused' ? -0.06 : 0)
    const steadyTargetOpen = Math.max(0.28, Math.min(0.98, baseOpen + sentimentOpenDelta))

    if (state === 'waking' && !reduced.matches) {
      const w = computeWakingCurve(wakeElapsedSec, steadyTargetOpen)
      uniforms.openness.value += (w.open - uniforms.openness.value) * 0.22
      uniforms.squeeze.value += (w.squeeze - uniforms.squeeze.value) * 0.22
      uniforms.pupilScale.value += (((d.pupil_scale ?? 1.0) * w.pupilBoost) - uniforms.pupilScale.value) * 0.12
      const sleepyGaze = new THREE.Vector2(pointerGaze[0] * 0.35, -(pointerGaze[1] * 0.35) + w.droopY)
      uniforms.gaze.value.lerp(sleepyGaze, 0.04)
    } else {
      const blink = reduced.matches || state === 'recovering' ? 0
        : computeSpontaneousBlink(t, uniforms.arousal.value, uniforms.focus.value)
      const effectiveOpen = steadyTargetOpen * (1.0 - 0.96 * blink)
      uniforms.openness.value += (effectiveOpen - uniforms.openness.value)
        * (state === 'recovering' || reduced.matches ? ease : blink > 0.1 ? 0.48 : 0.11)
      uniforms.squeeze.value += (0.0 - uniforms.squeeze.value) * 0.18
      uniforms.pupilScale.value += ((d.pupil_scale ?? 1.0) - uniforms.pupilScale.value) * 0.08

      const cue = props.observation && time - props.observation.at < 3500 ? props.observation : undefined
      // Lively micro-saccade subtle awareness scaled by neural arousal
      const microX = reduced.matches ? 0 : 0.018 * (0.4 + uniforms.arousal.value) * Math.sin(t * 2.3)
      const microY = reduced.matches ? 0 : 0.014 * (0.4 + uniforms.arousal.value) * Math.cos(t * 1.8)
      const gx = (cue?.x ?? pointerGaze[0]) + microX
      const gy = -(cue?.y ?? pointerGaze[1]) + microY
      uniforms.gaze.value.lerp(new THREE.Vector2(gx, gy), 0.07 + 0.04 * uniforms.focus.value)
    }

    const values =
      props.signal?.source === 'model-vector'
        ? vectorAt(props.signal.frames, props.playback?.present ? props.playback.at_ms : time - start)
        : []
    uniforms.vector.value.lerp(
      new THREE.Vector4(...([0, 1, 2, 3].map((i) => Math.tanh(values[i] || 0)) as [number, number, number, number])),
      0.1
    )

    eye.rotation.y = uniforms.gaze.value.x * 0.09
    eye.rotation.x = -uniforms.gaze.value.y * 0.06
    cornea.rotation.copy(eye.rotation)
  }

  if (!reduced.matches) {
    const sleepDamp = state === 'settling_sleep' ? 0.28 : state === 'rem_sleep' ? 0.55
      : 1 - uniforms.standby.value * 0.72
    const speed = (d.particle_speed ?? 0.25) * sleepDamp * (0.85 + 0.35 * uniforms.arousal.value)
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i], y = positions[i + 1], z = positions[i + 2]
      const r2 = x * x + y * y + z * z + 0.5
      const force = (0.012 * speed) / Math.pow(r2, 1.5)
      for (let j = 0; j < 3; j++) {
        const swirl = j === 0 ? -y : j === 1 ? x : 0
        velocities[i + j] +=
          (swirl * force + (homes[i + j] - positions[i + j]) * 0.025 - velocities[i + j] * 0.2) * dt
        positions[i + j] += velocities[i + j] * dt
      }
    }
    stars.geometry.attributes.position.needsUpdate = true
    stars.rotation.y += dt * 0.003 * sleepDamp
    nebula.rotation.y += dt * 0.001 * sleepDamp
  }

  renderer.render(scene, camera)

  if (time - reportAt > 15000) {
    const ctx = preview.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(renderer.domElement, 0, 0, 64, 64)
    const rgba = ctx.getImageData(0, 0, 64, 64).data
    const packed = new Uint8Array(4096)
    for (let i = 0; i < packed.length; i++) {
      packed[i] = ((rgba[i * 4] >> 6) << 4) | ((rgba[i * 4 + 1] >> 6) << 2) | (rgba[i * 4 + 2] >> 6)
    }
    emit('telemetry', {
      renderer: 'threejs-mutable-eye-v2',
      lifecycle_state: state,
      fps: Math.round(frames / Math.max(0.001, (time - reportAt) / 1000)),
      particles: positions.length / 3,
      spiral_count: Math.round(uniforms.spiralArms.value),
      source: props.signal?.source || 'ambient',
      webgl: true,
      reduced_motion: reduced.matches,
      preview: {
        width: 64,
        height: 64,
        encoding: 'rgb222-u8-base64',
        frame: btoa(String.fromCharCode(...packed)),
      },
    })
    frames = 0
    reportAt = time
  }
}

onMounted(() => {
  try {
    renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: 'low-power',
    })
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5))
    renderer.setClearColor(0x000000, 0)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.35
    host.value!.prepend(renderer.domElement)
    renderer.domElement.setAttribute('role', 'img')
    renderer.domElement.setAttribute(
      'aria-label',
      'W1z4rD living 3D crystal eye responding to sleep, REM, waking, neural sentiment, and self-redesign'
    )

    scene = new THREE.Scene()
    camera = new THREE.PerspectiveCamera(41, 1, 0.1, 30)
    camera.position.set(0, 0, 4.5)

    const universeMaterial = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      transparent: true,
      uniforms,
      vertexShader: vertex,
      fragmentShader: `
        precision highp float;
        varying vec3 localPosition;
        varying vec2 vScreen;
        uniform float time, nebula, hue, secondaryHue, standby;
        ${noise}
        void main() {
          float rScreen = length(vScreen);
          if (rScreen > 0.98) discard;
          float circleFade = smoothstep(0.98, 0.72, rScreen);

          vec3 p = normalize(localPosition);
          float cloud = fbm(p * 5.0 + vec3(time * 0.002, 0.0, 0.0));
          float dust = fbm(p * 12.0 + cloud * 3.0);
          float band = exp(-pow((p.y + 0.22 * p.x + 0.08 * sin(p.z * 9.0)) * 3.0, 2.0));

          vec3 nebulaTone = mix(hueColor(hue) * 0.18, hueColor(secondaryHue) * 0.18, dust);
          float dim = mix(1.0, 0.48, standby);
          vec3 cloudColor = nebulaTone * pow(cloud, 2.0) * band * nebula * 3.2 * dim;
          cloudColor += vec3(0.32, 0.20, 0.28) * pow(cloud * dust, 3.0) * nebula * 2.4 * dim;

          float alpha = clamp(length(cloudColor) * 2.2, 0.0, 1.0) * circleFade;
          gl_FragColor = vec4(pow(max(cloudColor, vec3(0.0)), vec3(0.65)), alpha);
        }
      `,
    })

    nebula = new THREE.Mesh(new THREE.SphereGeometry(9, 48, 32), universeMaterial)
    scene.add(nebula)

    const count = 2200
    positions = new Float32Array(count * 3)
    velocities = new Float32Array(count * 3)
    const colors = new Float32Array(count * 3)
    const sizes = new Float32Array(count)

    let seed = 1947
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
      return seed / 4294967296
    }

    for (let i = 0; i < count; i++) {
      const a = random() * Math.PI * 2
      const z = random() * 2 - 1
      const r = 3 + random() * 5
      const s = Math.sqrt(1 - z * z)
      positions.set([Math.cos(a) * s * r, z * r, Math.sin(a) * s * r], i * 3)
      const tint = random()
      colors.set(
        tint > 0.94 ? [1, 0.62, 0.8] : tint > 0.85 ? [0.65, 0.76, 1] : [0.85, 0.86, 0.92],
        i * 3
      )
      sizes[i] = random() > 0.98 ? 3.2 : 0.5 + random() * 1.3
    }

    homes = positions.slice()
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1))

    stars = new THREE.Points(
      geometry,
      new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        vertexColors: true,
        uniforms: { time: uniforms.time, sleepMode: uniforms.sleepMode, standby: uniforms.standby },
        vertexShader: `
          attribute float size;
          varying vec3 tint;
          varying float sparkle;
          varying vec2 vScreen;
          uniform float time, sleepMode, standby;
          void main() {
            tint = color;
            float pulse = sleepMode > 1.5 ? 0.5 : (0.7 + 0.3 * sin(time * 0.6 + position.x * 17.0));
            sparkle = pulse * mix(1.0, 0.45, standby);
            vec4 p = modelViewMatrix * vec4(position, 1.0);
            gl_Position = projectionMatrix * p;
            vScreen = gl_Position.xy / gl_Position.w;
            gl_PointSize = clamp(size * 8.0 / max(1.0, -p.z), 0.8, 8.0);
          }
        `,
        fragmentShader: `
          varying vec3 tint;
          varying float sparkle;
          varying vec2 vScreen;
          void main() {
            if (length(vScreen) > 0.96) discard;
            vec2 p = gl_PointCoord - 0.5;
            float d = length(p);
            float core = exp(-d * d * 65.0);
            float rays = exp(-abs(p.x) * 75.0 - abs(p.y) * 8.0) + exp(-abs(p.y) * 75.0 - abs(p.x) * 8.0);
            gl_FragColor = vec4(tint, (core + rays * 0.15) * sparkle);
          }
        `,
      })
    )
    scene.add(stars)

    eye = new THREE.Mesh(
      new THREE.SphereGeometry(1, 128, 80),
      new THREE.ShaderMaterial({
        uniforms,
        vertexShader: vertex,
        fragmentShader: irisFragment,
        transparent: true,
        side: THREE.FrontSide,
      })
    )
    eye.scale.set(1.18, 0.82, 0.48)
    eye.position.set(0, 0.31, 0.15)
    scene.add(eye)

    cornea = new THREE.Mesh(
      new THREE.SphereGeometry(1.035, 96, 64),
      new THREE.MeshPhysicalMaterial({
        color: 0xf4f0ff,
        roughness: 0.04,
        metalness: 0,
        transmission: 0.97,
        thickness: 0.09,
        ior: 1.376,
        clearcoat: 1,
        clearcoatRoughness: 0.02,
        transparent: true,
        opacity: 0.18,
        depthWrite: false,
      })
    )
    cornea.scale.copy(eye.scale)
    cornea.position.copy(eye.position)
    scene.add(cornea)

    const key = new THREE.DirectionalLight(0xf4e5ff, 3)
    key.position.set(-3, 4, 5)
    scene.add(key)
    const rim = new THREE.PointLight(0x829cff, 12)
    rim.position.set(2, -1, 2)
    scene.add(rim)
    scene.add(new THREE.AmbientLight(0x9b79c9, 0.7))

    const glass = new THREE.Mesh(
      new THREE.SphereGeometry(1.53, 96, 64),
      new THREE.MeshPhysicalMaterial({
        color: 0xded8f4,
        roughness: 0.045,
        transmission: 1,
        thickness: 0.1,
        ior: 1.46,
        clearcoat: 1,
        clearcoatRoughness: 0.03,
        transparent: true,
        opacity: 0.08,
        depthWrite: false,
      })
    )
    scene.add(glass)

    resize = new ResizeObserver(() => {
      if (!host.value) return
      const size = Math.max(1, Math.round(host.value.clientWidth))
      renderer?.setSize(size, size)
    })
    resize.observe(host.value!)
    association()
    reportAt = performance.now()
    frame = requestAnimationFrame(animate)
  } catch {
    error.value = '3D renderer unavailable on this device'
    emit('telemetry', { webgl: false, error: 'renderer unavailable' })
  }
})

onBeforeUnmount(() => {
  cancelAnimationFrame(frame)
  resize?.disconnect()
  pupilTexture?.dispose()
  scene?.traverse((object) => {
    const mesh = object as THREE.Mesh
    mesh.geometry?.dispose()
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    materials.forEach((material) => material?.dispose())
  })
  renderer?.dispose()
  renderer?.forceContextLoss()
  renderer?.domElement.remove()
})
</script>

<style scoped>
.eye {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  border-radius: 50%;
  clip-path: circle(50% at 50% 50%);
  -webkit-clip-path: circle(50% at 50% 50%);
  background: transparent !important;
  display: flex;
  align-items: center;
  justify-content: center;
  box-sizing: border-box;
  margin: 0 auto;
  transition: filter 0.6s ease;
}
.eye.eye-state-offline {
  filter: saturate(0.62) brightness(0.84);
}
.eye.eye-state-connecting { filter: saturate(0.78) brightness(0.94); }
@media (prefers-reduced-motion: reduce) {
  .eye { transition: none; }
}

.eye :deep(canvas),
.eye canvas {
  width: 100% !important;
  height: 100% !important;
  max-width: 100%;
  max-height: 100%;
  border-radius: 50%;
  clip-path: circle(50% at 50% 50%);
  -webkit-clip-path: circle(50% at 50% 50%);
  display: block;
  object-fit: cover;
  margin: 0 auto;
  background: transparent !important;
}

.signal-label {
  position: absolute;
  top: 5%;
  left: 19%;
  right: 19%;
  text-align: center;
  color: #cbbad188;
  font-size: clamp(0.38rem, 0.85cqw, 0.58rem);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  pointer-events: none;
}

.render-error {
  position: absolute;
  top: 30%;
  left: 15%;
  right: 15%;
  text-align: center;
  color: #eaa8d0;
  font-size: 0.8rem;
}
</style>
