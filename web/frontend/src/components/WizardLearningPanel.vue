<template>
  <details class="learning-panel" @toggle="onToggle">
    <summary>Foundation and real-video queue <span v-if="data">· {{ data.gate.ready ? 'scoped expansion' : 'foundation first' }}</span></summary>
    <div class="actions"><button type="button" @click="refresh(false)" :disabled="loading">Refresh</button><button type="button" @click="refresh(true)" :disabled="loading">Check RuView connection</button></div>
    <p role="status">{{ error || (loading ? 'Loading…' : data?.gate.scope) }}</p>
    <template v-if="data">
      <p>Real camera footage only. Cartoons, generated footage, animation, and staged fiction are excluded. Archive availability does not establish commercial-training rights.</p>
      <ul v-if="!data.gate.ready"><li v-for="reason in data.gate.blockers" :key="reason">{{ reason }}</li></ul>
      <p>Assessments: {{ Object.entries(data.assessment_cases).map(([name, count]) => `${name} ${count}`).join(' · ') }}</p>
      <details><summary>Core prerogatives</summary><ol><li v-for="item in data.manifest.prerogatives" :key="item.id">{{ item.meaning }} <small>{{ item.source }}</small></li></ol></details>
      <p>RuView: {{ data.ruview.connected === true ? 'connected · estimates are not ground truth' : data.ruview.connected === false ? data.ruview.error : 'not probed' }}</p>
      <ul class="queue"><li v-for="item in data.queue" :key="item.id">
        <strong>{{ item.payload.title || `${item.kind} input` }}</strong> · {{ item.state }}
        <span v-if="item.payload.rights?.license_id"> · {{ item.payload.rights.license_id }}</span>
        <a v-if="sourceUrl(item)" :href="sourceUrl(item)" target="_blank" rel="noopener noreferrer">Source</a>
        <p>{{ item.eligible ? 'Eligible for reviewed learning' : item.blockers.join(' ') }}</p>
      </li></ul>
      <p v-if="!data.queue.length">The queue is empty.</p>
    </template>
  </details>
</template>
<script setup lang="ts">
import { ref, watch, onBeforeUnmount } from 'vue'
const props = defineProps<{ brainId: string }>()
interface QueueItem { id: string; kind: string; state: string; eligible: boolean; blockers: string[]; payload: { title?: string; source_page?: string; rights?: { source_page?: string; license_id?: string } } }
interface LearningStatus { gate: { ready: boolean; blockers: string[]; scope: string }; assessment_cases: Record<string, number>; manifest: { prerogatives: { id: string; meaning: string; source: string }[] }; queue: QueueItem[]; ruview: { connected: boolean | null; error?: string } }
const data = ref<LearningStatus>(), loading = ref(false), error = ref(''), open = ref(false)
let controller: AbortController | undefined
function sourceUrl(item: QueueItem): string | undefined {
  const value = item.payload.rights?.source_page || item.payload.source_page
  try { if (value && new URL(value).protocol === 'https:') return value } catch { /* malformed provenance */ }
}
async function refresh(probe: boolean) {
  controller?.abort(); controller = new AbortController()
  const ownController = controller
  loading.value = true; error.value = ''
  try {
    const response = await fetch(`/api/wizard-chat/learning/${probe ? '?probe_ruview=1' : ''}`, { signal: ownController.signal })
    const value = await response.json()
    if (!response.ok) throw new Error(value.error || 'Learning status unavailable')
    if (!value.gate || !Array.isArray(value.queue)) throw new Error('Unexpected learning status response')
    data.value = value
  } catch (issue) { if (!ownController.signal.aborted) error.value = issue instanceof Error ? issue.message : String(issue) }
  finally { if (controller === ownController) loading.value = false }
}
function onToggle(event: Event) { open.value = (event.target as HTMLDetailsElement).open; if (open.value && !data.value) void refresh(false) }
watch(() => props.brainId, () => { data.value = undefined; if (open.value) void refresh(false) })
onBeforeUnmount(() => controller?.abort())
</script>
<style scoped>
.learning-panel { flex-shrink: 0; padding: .5rem 1rem; max-height: 40vh; overflow: auto; color: #d9c6e5; font-size: .74rem; border-top: 1px solid #bda4cd30; }
summary { cursor: pointer; } .actions { display: flex; gap: .5rem; margin-top: .5rem; }
button { background: #170f24; color: #f0dcef; border: 1px solid #75617e; border-radius: 12px; padding: .3rem .6rem; cursor: pointer; }
button:disabled { opacity: .5; } .queue { padding-left: 1rem; } .queue > li { margin: .75rem 0; } a { color: #b2bdff; margin-left: .5rem; } small { opacity: .6; }
</style>
