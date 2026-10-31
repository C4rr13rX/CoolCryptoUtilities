<template>
  <details class="silhouette-studio">
    <summary>Teach the eye a silhouette</summary>
    <p>Choose an isolated object, review its outline, and associate it with a concept and expression. Saved associations stay in this browser.</p>
    <div class="fields">
      <label>Object image <input type="file" accept="image/png,image/webp,image/jpeg" @change="load" /></label>
      <label>Background <select v-model="mode" @change="convert"><option value="alpha">Transparent cutout</option><option value="border">Flat background</option></select></label>
      <label>Threshold {{ threshold }} <input v-model.number="threshold" type="range" min="1" max="200" @input="convert" /></label>
      <label>Concept <input v-model="concept" maxlength="120" placeholder="e.g. moon" /></label>
      <label>Expression <select v-model="expression"><option value="affectionate-tired">Affectionate and tired</option><option value="amused">Nearly amused</option><option value="wonder">Wonder</option><option value="neutral">Neutral</option></select></label>
    </div>
    <svg v-if="silhouette" :viewBox="`0 0 ${silhouette.width} ${silhouette.height}`" role="img" aria-label="Silhouette to review"><path :d="silhouette.path" fill="currentColor" fill-rule="evenodd" /></svg>
    <div class="actions">
      <button type="button" :disabled="!silhouette || !concept.trim()" @click="confirm">Save reviewed association</button>
      <button type="button" :disabled="!silhouette" @click="downloadSvg">Download SVG</button>
      <button type="button" :disabled="!count" @click="exportExamples">Export confirmed examples</button>
      <button type="button" :disabled="!count" @click="forget">Forget saved associations</button>
    </div>
    <p role="status">{{ status }}</p>
  </details>
</template>
<script setup lang="ts">
import { ref } from 'vue'
import { foregroundMask, maskToSilhouette, type Silhouette } from '@/utils/silhouette'
import { ASSOCIATION_KEY, readAssociations, saveAssociation, type Expression, type EyeAssociation } from '@/utils/eyeAssociations'
const emit = defineEmits<{ changed: [values: EyeAssociation[]] }>()
const mode = ref<'alpha' | 'border'>('alpha'), threshold = ref(32), concept = ref(''), expression = ref<Expression>('affectionate-tired')
const silhouette = ref<Silhouette>(), status = ref('No image loaded.'), count = ref(readAssociations().length)
let pixels: ImageData | undefined, loadGeneration = 0
async function load(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0]
  const generation = ++loadGeneration
  pixels = undefined; silhouette.value = undefined
  if (!file) return
  if (!['image/png', 'image/webp', 'image/jpeg'].includes(file.type) || file.size > 12 * 1024 * 1024) {
    status.value = 'Choose a PNG, WebP, or JPEG smaller than 12 MB.'; return
  }
  try {
    const bitmap = await createImageBitmap(file)
    if (generation !== loadGeneration) { bitmap.close(); return }
    // Preserve original proportions using decoded file dimensions.
    const scale = Math.min(1, 256 / Math.max(bitmap.width, bitmap.height))
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(bitmap.width * scale)); canvas.height = Math.max(1, Math.round(bitmap.height * scale))
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height); bitmap.close()
    if (generation !== loadGeneration) return
    pixels = ctx.getImageData(0, 0, canvas.width, canvas.height); convert()
  } catch (error) { status.value = error instanceof Error ? error.message : 'Image could not be decoded.' }
}
function convert() {
  silhouette.value = undefined
  if (!pixels) return
  try {
    silhouette.value = maskToSilhouette(foregroundMask(pixels, mode.value, threshold.value), pixels.width, pixels.height)
    status.value = 'Review the outline before saving. Flat-background extraction needs a clear border around the object.'
  } catch (error) { status.value = error instanceof Error ? error.message : 'Outline extraction failed.' }
}
function confirm() {
  if (!silhouette.value) return
  try {
    const values = saveAssociation({ version: 1, concept: concept.value.trim(), expression: expression.value, silhouette: silhouette.value, confirmed_at: new Date().toISOString() })
    count.value = values.length; emit('changed', values)
    status.value = 'Saved. When this concept activates, the eye recalls its silhouette and expression. No model training was submitted.'
  } catch (error) { status.value = error instanceof Error ? error.message : 'Could not save association.' }
}
function download(text: string, type: string, name: string) {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const link = document.createElement('a'); link.href = url; link.download = name; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
function downloadSvg() { if (silhouette.value) download(silhouette.value.svg, 'image/svg+xml', 'wizard-silhouette.svg') }
function exportExamples() {
  const examples = readAssociations().map(value => JSON.stringify({
    version: 1, kind: 'confirmed-concept-silhouette', concept: value.concept,
    expression: value.expression, confirmed_at: value.confirmed_at,
    coordinate_space: 'silhouette-occupancy-4x4-v1', vector: value.silhouette.vector,
    path: value.silhouette.path, width: value.silhouette.width, height: value.silhouette.height,
  })).join('\n')
  download(examples + '\n', 'application/x-ndjson', 'wizard-silhouette-examples.jsonl')
}
function forget() {
  try { localStorage.removeItem(ASSOCIATION_KEY); count.value = 0; emit('changed', []); status.value = 'Saved associations cleared.' }
  catch { status.value = 'Browser storage could not be cleared.' }
}
</script>
<style scoped>
.silhouette-studio { padding: .5rem 1rem; color: #dac7e4; font-size: .75rem; border-top: 1px solid #bda4cd30; max-height: 38vh; overflow: auto; flex-shrink: 0; }
summary { cursor: pointer; }
.fields, .actions { display: flex; flex-wrap: wrap; gap: .6rem; }
label { display: grid; gap: .25rem; }
input, select, button { color: #f0dcef; background: #170f24; border: 1px solid #75617e; border-radius: 6px; padding: .3rem; max-width: 240px; }
button { cursor: pointer; } button:disabled { opacity: .45; }
svg { display: block; width: 120px; height: 120px; margin: .5rem 0; background: #ada1b31a; color: #e9d9ef; }
</style>
