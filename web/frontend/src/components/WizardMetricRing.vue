<template>
  <div class="metric-ring" aria-label="24 live engineering metrics">
    <div v-for="(metric, index) in metrics" :key="metric.name" class="metric" :class="metric.state" :style="position(index)" :title="`${metric.name}: ${metric.value}. ${metric.detail}`" :aria-label="`${metric.name}: ${metric.value}; ${metric.state}`">
      <span class="metric-name">{{ metric.name }}</span>
      <svg viewBox="0 0 64 64" aria-hidden="true"><path class="seal" :d="sigil(index).seal"/><path :d="paths?.[index] || sigil(index).rune"/><circle v-for="(point, i) in sigil(index).stars" :key="i" :cx="point[0]" :cy="point[1]" r="1.5"/></svg>
      <span class="metric-value">{{ metric.value }}</span>
    </div>
    <span class="metric-legend">BLUE healthy / available · PINK warning / unknown · RED failing</span>
  </div>
</template>
<script setup lang="ts">
import { computed, ref, onMounted, onBeforeUnmount, watch } from 'vue'
import { metricPosition as position, metricSigil as sigil } from '@/utils/metricSigils'
const props = defineProps<{ status: Record<string, any>; online: boolean | null; message?: Record<string, any>; amplitude: number; busy: boolean; brainId: string; paths?:string[] }>()
const learning = ref<any>(), updated = ref(0), now = ref(Date.now())
let timer: ReturnType<typeof setInterval>, controller: AbortController | undefined
async function refresh() {
  controller?.abort(); const own = controller = new AbortController()
  try {
    const response = await fetch('/api/wizard-chat/learning/', { signal: own.signal })
    if (!response.ok) throw new Error('Unavailable')
    const value = await response.json(); if (!own.signal.aborted) { learning.value = value; updated.value = Date.now() }
  } catch { if (!own.signal.aborted) learning.value = undefined }
}
watch(() => props.brainId, () => { learning.value = undefined; updated.value = 0; void refresh() })
onMounted(() => { void refresh(); timer = setInterval(() => { now.value = Date.now(); void refresh() }, 10000) })
onBeforeUnmount(() => { clearInterval(timer); controller?.abort() })
type State = 'blue' | 'pink' | 'red'
const metrics = computed(() => {
  const s = props.status, h = s.health || {}, b = s.brain || {}, m = props.message || {}, totals = s.multi_pool_stats?.totals || {}
  const number = (name: string, raw: unknown, unit = '', warn?: number, fail?: number) => {
    const valid = typeof raw === 'number' && Number.isFinite(raw)
    return { name, value: valid ? `${new Intl.NumberFormat('en', { maximumFractionDigits: 2, notation: Math.abs(raw as number) >= 10000 ? 'compact' : 'standard' }).format(raw as number)}${unit}` : '—', state: (!valid ? 'pink' : fail != null && (raw as number) >= fail ? 'red' : warn != null && (raw as number) >= warn ? 'pink' : 'blue') as State,
      detail: valid ? warn == null ? 'Color indicates telemetry availability, not model quality.' : `Local display thresholds: warning ≥ ${warn}${unit}, failing ≥ ${fail}${unit}.` : 'Not reported by this brain; no value is fabricated.' }
  }
  const flag = (name: string, value: string, state: State, detail: string) => ({ name, value, state, detail })
  const gate = learning.value?.gate, cases = learning.value?.assessment_cases
  return [
    flag('Node availability', props.online == null ? 'Checking' : props.online ? 'Online' : 'Offline', props.online == null ? 'pink' : props.online ? 'blue' : 'red', 'Current node health poll.'),
    number('Node uptime', h.uptime_secs, ' s'),
    flag('Inference activity', props.busy ? 'Thinking' : 'Idle', 'blue', 'Local request activity; idle is normal.'),
    number('Inference latency', m.latency_ms ?? m.elapsed_ms, ' ms', 5000, 15000),
    number('Output tokens', m.output_tokens ?? m.usage?.output_tokens),
    number('Token throughput', m.tokens_per_second, ' /s'),
    number('Error rate', h.error_rate_percent, '%', 1, 5),
    number('CPU utilization', h.cpu_percent, '%', 80, 95),
    number('Memory utilization', h.memory_percent, '%', 80, 95),
    number('GPU utilization', h.gpu_percent, '%', 90, 99),
    number('Neural pools', totals.pool_count),
    number('Stored atoms', totals.atoms),
    number('Stored concepts', totals.concepts),
    number('Excitatory synapses', totals.exc_synapses),
    number('Inhibitory synapses', totals.inh_synapses),
    number('Cross-pool edges', totals.cross_pool_edges ?? b.multi_pool?.cross_edges),
    number('Activated concepts', Array.isArray(m.concepts) ? m.concepts.length : undefined),
    flag('Response confidence', m.confidence_tier || '—', m.confidence_tier === 'high' ? 'blue' : m.confidence_tier === 'error' ? 'red' : 'pink', 'Model-reported tier, not a calibrated accuracy measurement.'),
    number('Reasoning hops', m.hops),
    number('Vector dimensions', m.eye_signal?.frames?.[0]?.vector?.length),
    number('Voice amplitude', Math.round(props.amplitude * 100), '%'),
    flag('Foundation admission', gate ? gate.ready ? 'Passed' : 'Locked' : '—', gate?.ready ? 'blue' : 'pink', 'All required assessments must pass for this checkpoint. Locked is a policy state, not proof of model failure.'),
    number('Assessment coverage', cases ? Object.values(cases).reduce((sum: number, value) => sum + Number(value), 0) : undefined, ' cases'),
    number('Queue telemetry age', updated.value ? Math.max(0, Math.round((now.value - updated.value) / 1000)) : undefined, ' s', 30, 60),
  ]
})
</script>
<style scoped>
.metric-ring { position: absolute; inset: 0; pointer-events: none; border-radius: 50%; }
.metric-ring::before { content: ''; position: absolute; inset: 6%; border: 1px solid #ba8fdc18; border-radius: 50%; }
.metric { position: absolute; width: 10%; height: clamp(20px, 3.5cqw, 30px); transform: translate(-50%, -50%); display: flex; align-items: center; justify-content: center; text-align: center; min-width: 0; pointer-events: auto; color: #7eafff; --glow: #4385ff55; }
.pink { color: #f39acb; --glow: #ff62bd44; } .red { color: #ff6579; --glow: #ff254866; }
.metric-name { position:absolute;bottom:calc(100% + 3px);color: #e4deec; font-size: clamp(6px, .85cqw, 10px); line-height: 1.15; width: 100%; }
svg { color: inherit; width: 100%; height: 100%; fill: none; stroke: currentColor; stroke-width: 1.15; stroke-linecap: round; stroke-linejoin: round; filter: drop-shadow(0 0 6px var(--glow)); transition: color .6s; } .seal { opacity: .5; stroke-width: .65; }
.metric-value { position:absolute;top:calc(100% + 3px);font: 500 clamp(8px, 1cqw, 11px) ui-monospace, monospace; }
.metric-legend { position: absolute; bottom: -9px; width: 100%; text-align: center; font-size: clamp(6px, .8cqw, 9px); color: #b3a5c6; }
</style>
