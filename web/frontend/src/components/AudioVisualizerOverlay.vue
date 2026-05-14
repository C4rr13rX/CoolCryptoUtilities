<template>
  <!-- Translucent click-through overlay that modulates with the audio
       output from the wizard chat.  Renders inside the chat area as
       an absolutely-positioned canvas; `pointer-events: none` lets
       clicks pass through to messages underneath.

       Effect: a soft white luminance that brightens with overall
       loudness and ripples in horizontal bands keyed to the frequency
       spectrum — like sound rising up through fog.  Stays subtle when
       no audio is playing; never blocks user interaction. -->
  <canvas
    ref="canvasEl"
    class="audio-viz-overlay"
    aria-hidden="true"
  />
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'

// One shared AudioContext + analyser across the whole chat session.
// Every <audio> element we attach to plumbs into this analyser via a
// MediaElementSource; the canvas reads it on every animation frame.
let audioCtx:    AudioContext | null = null
let analyser:    AnalyserNode | null = null
let mergerGain:  GainNode | null = null
const attached:  WeakSet<HTMLMediaElement> = new WeakSet()

const canvasEl = ref<HTMLCanvasElement | null>(null)
let rafId = 0
let resizeObs: ResizeObserver | null = null
let mutationObs: MutationObserver | null = null
let scanIv: number | null = null

/** Ensure the audio graph exists.  Lazy: only built when the first
 *  audio element is attached, because creating an AudioContext too
 *  early can trip browser autoplay-policy warnings. */
function ensureAudioGraph(): boolean {
  if (audioCtx && analyser) return true
  try {
    // @ts-ignore - webkit fallback
    const Ctx = window.AudioContext || (window as any).webkitAudioContext
    if (!Ctx) return false
    audioCtx = new Ctx()
    mergerGain = audioCtx.createGain()
    mergerGain.gain.value = 1.0
    analyser = audioCtx.createAnalyser()
    analyser.fftSize = 512                  // 256-bin spectrum, cheap + smooth
    analyser.smoothingTimeConstant = 0.7
    mergerGain.connect(analyser)
    analyser.connect(audioCtx.destination)
    return true
  } catch {
    audioCtx = null; analyser = null; mergerGain = null
    return false
  }
}

/** Attach an <audio>/<video> element to the shared analyser so its
 *  output drives the overlay.  Idempotent — each element wired once. */
function attachAudioElement(el: HTMLMediaElement) {
  if (!ensureAudioGraph() || !audioCtx || !mergerGain) return
  if (attached.has(el)) return
  try {
    // Some browsers throw if the element was already wired to another
    // graph; swallow and skip if so.
    const src = audioCtx.createMediaElementSource(el)
    src.connect(mergerGain)
    attached.add(el)
  } catch { /* element already in another graph — leave it alone */ }
  // Resume context on play (browsers gate autoplay until user gesture).
  const resume = () => audioCtx?.resume().catch(() => {})
  el.addEventListener('play', resume, { passive: true })
}

/** Scan the page for new audio/video elements and attach them. */
function scanForMedia() {
  const els = document.querySelectorAll<HTMLMediaElement>(
    'audio, video'
  )
  els.forEach(attachAudioElement)
}

/** Draw one frame: read spectrum + RMS, render translucent bands.
 *  Heuristic: split the FFT into 5 vertical bands; each band's amplitude
 *  modulates a horizontally-spanning soft-edged rectangle that fades
 *  from the bottom up.  Overall RMS controls global opacity so the
 *  effect goes invisible when no sound is playing. */
function drawFrame() {
  rafId = requestAnimationFrame(drawFrame)
  const canvas = canvasEl.value
  if (!canvas || !analyser) {
    // No audio yet — clear and bail.
    if (canvas) {
      const ctx = canvas.getContext('2d')
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
    return
  }
  const bins = analyser.frequencyBinCount
  const freq = new Uint8Array(bins)
  analyser.getByteFrequencyData(freq)

  // Per-band amplitude (5 bands, low→high frequency).
  const BANDS = 5
  const bandAmp: number[] = []
  for (let b = 0; b < BANDS; b++) {
    const lo = Math.floor((b / BANDS) * bins)
    const hi = Math.floor(((b + 1) / BANDS) * bins)
    let sum = 0
    for (let i = lo; i < hi; i++) sum += freq[i]
    bandAmp.push(sum / Math.max(1, (hi - lo) * 255))   // normalize 0..1
  }
  // Global RMS for overall opacity.
  let rms = 0
  for (let i = 0; i < bins; i++) rms += freq[i] * freq[i]
  rms = Math.sqrt(rms / bins) / 255

  const W = canvas.width, H = canvas.height
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, W, H)
  if (rms < 0.01) return   // silence — no overlay

  // Soft base wash that brightens with loudness.
  const baseAlpha = Math.min(0.18, rms * 0.32)
  ctx.fillStyle = `rgba(255, 255, 255, ${baseAlpha})`
  ctx.fillRect(0, 0, W, H)

  // Vertical band-amplitude curtains.  Drawn bottom-up with a soft
  // gradient so the effect feels like luminous mist responding to
  // the voice's spectral shape.
  for (let b = 0; b < BANDS; b++) {
    const amp = bandAmp[b]
    if (amp < 0.05) continue
    // Distribute bands across vertical thirds: lows at the bottom,
    // mids in the middle, highs at the top.  Each band occupies a
    // wide horizontal stripe but its alpha is the band amplitude.
    const yMid = H * (1 - (b + 0.5) / BANDS)
    const bandH = H * 0.45
    const grad = ctx.createLinearGradient(0, yMid - bandH/2, 0, yMid + bandH/2)
    const alpha = Math.min(0.35, amp * 0.55)
    grad.addColorStop(0,   `rgba(255, 255, 255, 0)`)
    grad.addColorStop(0.5, `rgba(255, 255, 255, ${alpha})`)
    grad.addColorStop(1,   `rgba(255, 255, 255, 0)`)
    ctx.fillStyle = grad
    ctx.fillRect(0, yMid - bandH/2, W, bandH)
  }

  // Top-down "halo" pulse on the strongest band — "god through sky".
  const peakBand = bandAmp.indexOf(Math.max(...bandAmp))
  if (bandAmp[peakBand] > 0.25) {
    const haloGrad = ctx.createRadialGradient(W/2, -H*0.1, 0, W/2, -H*0.1, H*1.1)
    haloGrad.addColorStop(0,   `rgba(255,255,255, ${Math.min(0.32, bandAmp[peakBand] * 0.5)})`)
    haloGrad.addColorStop(0.6, `rgba(255,255,255, ${Math.min(0.12, bandAmp[peakBand] * 0.18)})`)
    haloGrad.addColorStop(1,   `rgba(255,255,255, 0)`)
    ctx.fillStyle = haloGrad
    ctx.fillRect(0, 0, W, H)
  }
}

function resizeCanvas() {
  const canvas = canvasEl.value
  if (!canvas) return
  const parent = canvas.parentElement
  if (!parent) return
  const dpr = window.devicePixelRatio || 1
  canvas.width  = Math.floor(parent.clientWidth  * dpr)
  canvas.height = Math.floor(parent.clientHeight * dpr)
  canvas.style.width  = parent.clientWidth + 'px'
  canvas.style.height = parent.clientHeight + 'px'
}

onMounted(() => {
  resizeCanvas()
  resizeObs = new ResizeObserver(resizeCanvas)
  if (canvasEl.value?.parentElement)
    resizeObs.observe(canvasEl.value.parentElement)
  // Find any existing media + watch for new ones added by chat messages.
  scanForMedia()
  mutationObs = new MutationObserver(() => scanForMedia())
  mutationObs.observe(document.body, { childList: true, subtree: true })
  // Also poll every 2s as a safety net (some frameworks reuse nodes).
  scanIv = window.setInterval(scanForMedia, 2000)
  drawFrame()
})

onBeforeUnmount(() => {
  if (rafId) cancelAnimationFrame(rafId)
  if (scanIv) clearInterval(scanIv)
  resizeObs?.disconnect()
  mutationObs?.disconnect()
})
</script>

<style scoped>
.audio-viz-overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;      /* click-through */
  z-index: 5;
  mix-blend-mode: screen;    /* additive feel against the dark chat */
  user-select: none;
}
</style>
