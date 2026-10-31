<template>
  <details class="observation">
    <summary>Observation <span>{{ active ? '· active locally' : '· off' }}</span></summary>
    <div class="controls">
      <button type="button" @click="start('camera')" :disabled="!!streams.camera || pending.camera">{{ pending.camera ? 'Requesting webcam…' : 'Enable webcam' }}</button>
      <button type="button" @click="start('screen')" :disabled="!!streams.screen || pending.screen">{{ pending.screen ? 'Requesting screen…' : 'Share screen' }}</button>
      <button type="button" @click="stop">Stop observation</button>
      <label>Screen detail <select v-model="detail"><option value="auto">Adaptive · preserve possible text</option><option value="original">Original detail</option><option value="1280">1280 px</option><option value="640">640 px</option></select></label>
      <label><input v-model="interactions" type="checkbox" /> Observe app interaction activity</label>
    </div>
    <p role="status">{{ status }} {{ error }}</p>
    <p v-if="interactions">{{ counts.keys }} key events · {{ counts.clicks }} clicks. Key contents and sensitive-field activity are excluded.</p>
    <div class="previews">
      <figure v-show="!!streams.camera"><canvas ref="cameraCanvas" aria-label="Local reduced-color webcam preview" /><figcaption>Webcam · {{ dimensions.camera }} · motion signal only</figcaption></figure>
      <figure v-show="!!streams.screen"><canvas ref="screenCanvas" aria-label="Local reduced-color screen preview" /><figcaption>Screen · {{ dimensions.screen }} · check text readability here</figcaption></figure>
    </div>
  </details>
</template>
<script setup lang="ts">
import { ref, reactive, computed, onMounted, onBeforeUnmount } from 'vue'
import { fitFrame, mayContainText, motionCue, quantize64, type ObservationCue } from '@/utils/observation'
type Kind = 'camera' | 'screen'
const emit = defineEmits<{ cue: [value: ObservationCue | undefined] }>()
const streams = reactive<Partial<Record<Kind, MediaStream>>>({}), pending = reactive({ camera: false, screen: false })
const cameraCanvas = ref<HTMLCanvasElement>(), screenCanvas = ref<HTMLCanvasElement>()
const detail = ref('auto'), interactions = ref(false), error = ref('')
const dimensions = reactive({ camera: '', screen: '' }), counts = reactive({ keys: 0, clicks: 0 })
const active = computed(() => !!streams.camera || !!streams.screen || interactions.value)
const status = computed(() => active.value ? 'Local observation · 2 fps / 64 colors · no recording or training upload.' : 'Observation off.')
const videos: Partial<Record<Kind, HTMLVideoElement>> = {}
let timer: ReturnType<typeof setInterval> | undefined, generation = 0, previousCamera: Uint8ClampedArray | undefined
let lastPointer = -Infinity, lastPointerSample = 0, sampleNumber = 0, needsDetail = true
function sample() {
  sampleNumber++
  for (const kind of ['camera', 'screen'] as const) {
    const video = videos[kind], canvas = kind === 'camera' ? cameraCanvas.value : screenCanvas.value
    if (!video || video.readyState < 2 || !canvas || !video.videoWidth) continue
    if (kind === 'screen' && detail.value === 'auto' && sampleNumber % 4 === 1) {
      const probe = document.createElement('canvas'); [probe.width, probe.height] = fitFrame(video.videoWidth, video.videoHeight, 960)
      const ctx = probe.getContext('2d', { willReadFrequently: true })!
      ctx.drawImage(video, 0, 0, probe.width, probe.height)
      needsDetail = mayContainText(ctx.getImageData(0, 0, probe.width, probe.height))
    }
    const maxWidth = kind === 'camera' ? 160 : detail.value === 'original' || detail.value === 'auto' && needsDetail
      ? video.videoWidth : detail.value === 'auto' ? 640 : Number(detail.value)
    const [width, height] = fitFrame(video.videoWidth, video.videoHeight, maxWidth)
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height }
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(video, 0, 0, width, height)
    const image = ctx.getImageData(0, 0, width, height)
    if (kind === 'camera') {
      const cue = motionCue(image.data, previousCamera, width, height)
      previousCamera = new Uint8ClampedArray(image.data)
      if (cue && performance.now() - lastPointer > 2000) emit('cue', cue)
    }
    quantize64(image.data); ctx.putImageData(image, 0, 0)
    dimensions[kind] = `${width} × ${height}`
  }
}
function release(kind: Kind) {
  const stream = streams[kind], video = videos[kind]
  delete streams[kind]; delete videos[kind]
  stream?.getTracks().forEach(track => track.stop())
  if (video) { video.pause(); video.srcObject = null }
  const canvas = kind === 'camera' ? cameraCanvas.value : screenCanvas.value
  canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
  if (kind === 'camera') previousCamera = undefined
  if (!Object.keys(streams).length && timer) { clearInterval(timer); timer = undefined; emit('cue', undefined) }
}
async function start(kind: Kind) {
  if (pending[kind] || streams[kind]) return
  const requestGeneration = generation
  pending[kind] = true; error.value = ''
  let stream: MediaStream | undefined
  try {
    if (!navigator.mediaDevices) throw new Error('Capture needs a secure browser context.')
    stream = kind === 'camera'
      ? await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 320 }, height: { ideal: 180 }, frameRate: { ideal: 2, max: 2 } }, audio: false })
      : await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: { ideal: 2, max: 2 } }, audio: false })
    if (requestGeneration !== generation) { stream.getTracks().forEach(track => track.stop()); return }
    streams[kind] = stream
    const video = document.createElement('video'); video.muted = true; video.srcObject = stream; videos[kind] = video
    const ownedStream = stream
    stream.getVideoTracks()[0].addEventListener('ended', () => { if (streams[kind] === ownedStream) release(kind) }, { once: true })
    await video.play()
    if (requestGeneration !== generation || streams[kind] !== ownedStream) return
    sampleNumber = 0; sample()
    if (!timer) timer = setInterval(sample, 500)
  } catch (issue) {
    if (stream && streams[kind] === stream) release(kind)
    else stream?.getTracks().forEach(track => track.stop())
    if (requestGeneration === generation) error.value = issue instanceof Error ? issue.message : String(issue)
  } finally { if (requestGeneration === generation) pending[kind] = false }
}
function sensitive(event: Event): boolean {
  const element = event.target instanceof Element ? event.target : undefined
  return !!element?.closest('input[type="password"], [data-private], [autocomplete="one-time-code"], [autocomplete^="cc-"], [autocomplete="current-password"], [autocomplete="new-password"]')
}
function pointer(event: PointerEvent) {
  if (!interactions.value || sensitive(event) || performance.now() - lastPointerSample < 100) return
  lastPointer = lastPointerSample = performance.now()
  emit('cue', { x: event.clientX / Math.max(1, innerWidth) - .5, y: event.clientY / Math.max(1, innerHeight) - .5, activity: .2, source: 'pointer', at: lastPointer })
}
function click(event: MouseEvent) { if (interactions.value && !sensitive(event)) counts.clicks++ }
function key(event: KeyboardEvent) { if (interactions.value && !sensitive(event) && !event.repeat) counts.keys++ }
function stop() {
  generation++; release('camera'); release('screen'); interactions.value = false
  pending.camera = pending.screen = false; counts.keys = counts.clicks = 0; error.value = ''; emit('cue', undefined)
}
onMounted(() => { window.addEventListener('pointermove', pointer, { passive: true }); window.addEventListener('click', click); window.addEventListener('keydown', key) })
onBeforeUnmount(() => { stop(); window.removeEventListener('pointermove', pointer); window.removeEventListener('click', click); window.removeEventListener('keydown', key) })
</script>
<style scoped>
.observation { padding: .5rem 1rem; font-size: .72rem; color: #cbb8d4; max-height: 35vh; overflow: auto; flex-shrink: 0; }
summary { cursor: pointer; } summary span { opacity: .65; }
.controls, .previews { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin-top: .5rem; }
button, select { background: #170f24; border: 1px solid #75617e; border-radius: 12px; padding: .3rem .6rem; color: #f0dcef; cursor: pointer; }
button:disabled { opacity: .5; } label { display: inline-flex; gap: .3rem; align-items: center; }
figure { margin: 0; } canvas { width: min(100%, 400px); max-height: 160px; object-fit: contain; border-radius: 8px; }
figcaption { font-size: .65rem; }
</style>
