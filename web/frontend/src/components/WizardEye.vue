<template>
  <div ref="host" class="eye" @pointermove="look" @pointerleave="gaze = [0,0]">
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

const props = defineProps<{
  concepts: string[];
  messageId: string;
  busy: boolean;
  amplitude: number;
  signal?: EyeSignal;
  playback?: { at_ms: number; present: boolean };
  association?: EyeAssociation;
  observation?: ObservationCue;
  design?: Record<string, number>;
}>()

const emit = defineEmits<{ telemetry: [value: Record<string, unknown>] }>()
const host = ref<HTMLDivElement>()
const error = ref('')

const signalLabel = computed(() =>
  props.signal?.source === 'model-vector'
    ? 'Model vector trajectory'
    : props.association
    ? `Recalled silhouette · ${props.association.concept}`
    : props.concepts.length
    ? 'Concept signature · snapshot'
    : 'Ambient visual · awaiting concepts'
)

let gaze = [0, 0]
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
  hue: { value: 0.79 },
  openness: { value: 0.65 },
  tightness: { value: 0.16 },
  nebula: { value: 0.8 },
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
  return mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), 0.75);
}
`

const irisFragment = `
precision highp float;
varying vec3 localPosition;
varying vec3 normalView;
varying vec3 viewPosition;

uniform float time, voice, hue, openness, tightness, hasPupil;
uniform vec2 gaze;
uniform vec4 vector;
uniform sampler2D pupil;

${noise}

void main() {
  vec2 p = localPosition.xy;
  
  // Natural anatomical almond eyelid curve
  float lidProfile = pow(max(0.0, 1.0 - p.x * p.x * 1.05), 0.72);
  float upperLid = (0.08 + 0.72 * openness) * lidProfile;
  float lowerLid = (0.06 + 0.45 * openness) * lidProfile;
  
  // Asymmetrical realistic eye opening (upper lid moves more than lower)
  float edgeDist = p.y > 0.0 ? (upperLid - p.y) : (p.y + lowerLid);
  if (edgeDist < 0.0) discard;
  
  // Gaze parallax offset
  vec2 eyeCenter = vec2(gaze.x * 0.22, gaze.y * 0.14);
  vec2 irisCoord = p - eyeCenter;
  irisCoord.y *= 1.02; // slight vertical perspective
  
  float r = length(irisCoord);
  float a = atan(irisCoord.y, irisCoord.x);
  float irisRadius = 0.58;
  
  // ── Realistic Iris Anatomy ──
  // 1. Multi-frequency radial stromal fibers
  float fiberNoise = fbm(vec3(cos(a * 2.0) * 16.0, sin(a * 2.0) * 16.0, r * 24.0));
  float fineStriae = sin(a * 190.0 + noise(vec3(a * 14.0, r * 28.0, 2.5)) * 6.0 + r * 30.0);
  float radialRibs = sin(a * 42.0 + fiberNoise * 4.0);
  
  // 2. Pupillary and ciliary zone collarette (wavy boundary around r=0.28)
  float collaretteR = 0.28 + 0.03 * sin(a * 7.0 + fiberNoise * 3.0);
  float collaretteRing = smoothstep(0.04, 0.0, abs(r - collaretteR));
  
  // 3. Crypts of Fuchs (natural depressions/furrows in the stroma)
  float cryptMask = smoothstep(0.65, 0.95, fbm(vec3(cos(a * 5.0) * 8.0, sin(a * 5.0) * 8.0, r * 14.0)));
  
  // 4. Base pigment composition
  vec3 baseTone = hueColor(hue);
  vec3 darkPigment = mix(vec3(0.04, 0.025, 0.05), baseTone * 0.22, fiberNoise);
  vec3 vibrantPigment = mix(baseTone * 0.65, vec3(0.85, 0.78, 0.95), fineStriae * 0.35 + 0.5);
  vec3 irisColor = mix(darkPigment, vibrantPigment, smoothstep(0.18, 0.45, r));
  irisColor += vec3(0.25, 0.2, 0.3) * collaretteRing * 0.4;
  irisColor = mix(irisColor, darkPigment * 0.6, cryptMask * 0.5);
  
  // 5. Deep Limbal Ring (natural dark outer rim)
  float limbusInner = irisRadius - 0.07;
  float limbusOuter = irisRadius + 0.02;
  float limbus = smoothstep(limbusOuter, limbusInner, r);
  float limbusDarkRim = smoothstep(limbusInner, irisRadius, r) * (1.0 - smoothstep(irisRadius, limbusOuter, r));
  irisColor = mix(irisColor * 0.35, irisColor, 1.0 - limbusDarkRim * 0.75);
  
  // 6. Logarithmic vector spiral (subtly woven into the iris fibers)
  float deformation = 0.04 * (vector.x * sin(a * 2.0) + vector.y * cos(a * 3.0));
  float phase = a - log(max(0.025, r) / 0.025) / max(0.05, tightness) - time * 0.03 + deformation;
  float angularDist = abs(atan(sin(phase), cos(phase)));
  float spiral = exp(-pow(angularDist * max(0.07, r) * 115.0, 2.0))
               * smoothstep(0.11, 0.17, r)
               * (1.0 - smoothstep(0.50, 0.57, r));
  vec3 spiralColor = mix(vec3(0.92, 0.65, 0.82), vec3(0.68, 0.82, 1.0), 0.5 + 0.5 * sin(log(max(0.02, r)) * 4.0));
  irisColor += spiralColor * spiral * 1.5;
  
  // ── Realistic Sclera with Micro-capillaries ──
  vec3 scleraBase = vec3(0.92, 0.91, 0.94);
  float scleraShadow = smoothstep(0.0, 0.35, edgeDist); // shadow cast by lids
  float scleraTexture = fbm(vec3(p * 8.0, 1.0));
  vec3 sclera = mix(scleraBase * 0.78, scleraBase, scleraShadow * 0.6 + scleraTexture * 0.4);
  
  // Fine micro-vascular capillary network
  float capNoise = fbm(vec3(p * 22.0, 4.0));
  float vessel = pow(max(0.0, sin(a * 18.0 + capNoise * 9.0 + r * 22.0)), 28.0) * smoothstep(0.52, 0.95, r);
  sclera = mix(sclera, vec3(0.72, 0.22, 0.28), vessel * 0.45);
  
  // Blend sclera into iris via limbal gradient
  vec3 eyeColor = mix(sclera, irisColor, limbus);
  
  // ── Pupil Dynamics ──
  float pupilRadius = 0.13 + voice * 0.022 + 0.007 * sin(time * 0.4);
  float pupilFeather = 0.009;
  float pupilMask = 1.0 - smoothstep(pupilRadius - pupilFeather, pupilRadius + pupilFeather, r);
  
  if (hasPupil > 0.5) {
    vec2 uv = irisCoord / 0.48 + 0.5;
    pupilMask = texture2D(pupil, uv).a * step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
  }
  
  // Pupil depth and warm sphincter tone
  eyeColor = mix(eyeColor, vec3(0.005, 0.003, 0.008), pupilMask);
  
  // ── Lighting & Cornea Wet Specular Reflections ──
  vec3 n = normalize(normalView);
  vec3 v = normalize(viewPosition);
  vec3 light1 = normalize(vec3(-0.55, 0.75, 1.8)); // primary key light
  vec3 light2 = normalize(vec3(0.65, -0.4, 1.2));  // soft bounce fill
  
  float diff = 0.35 + 0.65 * max(0.0, dot(n, light1));
  eyeColor *= diff;
  
  // Dual highlights (primary crisp highlight + soft secondary reflection)
  vec3 h1 = normalize(light1 + v);
  float spec1 = pow(max(0.0, dot(n, h1)), 120.0);
  vec3 h2 = normalize(light2 + v);
  float spec2 = pow(max(0.0, dot(n, h2)), 45.0);
  eyeColor += vec3(0.95, 0.98, 1.0) * spec1 * 0.85;
  eyeColor += vec3(0.6, 0.7, 0.9) * spec2 * 0.25;
  
  // Eyelid margin wet line and subtle shadow
  eyeColor *= smoothstep(0.0, 0.035, edgeDist);
  eyeColor += vec3(0.38, 0.24, 0.32) * exp(-abs(edgeDist - 0.012) * 110.0) * 0.3;
  
  // Output with organic tonemapping & smooth edge anti-aliasing
  gl_FragColor = vec4(pow(max(eyeColor, vec3(0.0)), vec3(0.72)), smoothstep(0.0, 0.012, edgeDist));
}
`

function look(event: PointerEvent) {
  const rect = host.value!.getBoundingClientRect()
  gaze = [
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

function animate(time: number) {
  frame = requestAnimationFrame(animate)
  if (!renderer || time - last < (reduced.matches ? 200 : 33) || document.hidden) return
  const dt = Math.min(0.05, (time - last) / 1000 || 0.033)
  last = time
  frames++

  const t = reduced.matches ? 0 : time / 1000
  const d = props.design || {}

  uniforms.time.value = t
  uniforms.voice.value = props.amplitude
  uniforms.hue.value = (d.hue ?? 285) / 360
  uniforms.tightness.value = d.spiral_tightness ?? 0.16
  uniforms.nebula.value = d.nebula_strength ?? 0.8

  const expression = props.association?.expression
  const targetOpen =
    d.openness ??
    (props.busy ? 0.82 : expression === 'wonder' ? 0.93 : expression === 'amused' ? 0.58 : 0.67)
  uniforms.openness.value += (targetOpen - uniforms.openness.value) * 0.06

  const cue = props.observation && time - props.observation.at < 3500 ? props.observation : undefined
  uniforms.gaze.value.lerp(new THREE.Vector2(cue?.x ?? gaze[0], -(cue?.y ?? gaze[1])), 0.06)

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

  if (!reduced.matches) {
    const speed = d.particle_speed ?? 0.25
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
    stars.rotation.y = t * 0.003
    nebula.rotation.y = t * 0.001
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
      renderer: 'threejs-mutable-eye-v1',
      fps: Math.round(frames / Math.max(0.001, (time - reportAt) / 1000)),
      particles: positions.length / 3,
      spiral_count: 1,
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
    // Perfectly transparent clear color so the WebGL viewport has no box edges
    renderer.setClearColor(0x000000, 0)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.35
    host.value!.prepend(renderer.domElement)
    renderer.domElement.setAttribute('role', 'img')
    renderer.domElement.setAttribute(
      'aria-label',
      'Three dimensional crystal eye with one logarithmic spiral and a particle universe'
    )

    scene = new THREE.Scene()
    camera = new THREE.PerspectiveCamera(41, 1, 0.1, 30)
    camera.position.set(0, 0, 4.5)

    // Cosmic background that is strictly inscribed inside the circular orb
    // with smooth edge falloff and discard outside rScreen > 0.98, eliminating
    // all square boundary corners.
    const universeMaterial = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      transparent: true,
      uniforms,
      vertexShader: vertex,
      fragmentShader: `
        precision highp float;
        varying vec3 localPosition;
        varying vec2 vScreen;
        uniform float time, nebula;
        ${noise}
        void main() {
          float rScreen = length(vScreen);
          if (rScreen > 0.98) discard;
          float circleFade = smoothstep(0.98, 0.72, rScreen);
          
          vec3 p = normalize(localPosition);
          float cloud = fbm(p * 5.0 + vec3(time * 0.002, 0.0, 0.0));
          float dust = fbm(p * 12.0 + cloud * 3.0);
          float band = exp(-pow((p.y + 0.22 * p.x + 0.08 * sin(p.z * 9.0)) * 3.0, 2.0));
          
          vec3 nebulaTone = mix(vec3(0.14, 0.02, 0.16), vec3(0.025, 0.055, 0.19), dust);
          vec3 cloudColor = nebulaTone * pow(cloud, 2.0) * band * nebula * 3.5;
          cloudColor += vec3(0.35, 0.22, 0.28) * pow(cloud * dust, 3.0) * nebula * 2.8;
          
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
        uniforms: { time: uniforms.time },
        vertexShader: `
          attribute float size;
          varying vec3 tint;
          varying float sparkle;
          varying vec2 vScreen;
          uniform float time;
          void main() {
            tint = color;
            sparkle = 0.7 + 0.3 * sin(time * 0.6 + position.x * 17.0);
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

    // Eye mesh with realistic shader
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

    // Cornea lens with physical refraction, Fresnel, and clearcoat
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
      const size = Math.max(1, Math.round(host.value!.clientWidth))
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
  bottom: 7%;
  left: 19%;
  right: 19%;
  text-align: center;
  color: #cbbad17a;
  font-size: clamp(0.35rem, 0.85cqw, 0.55rem);
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
