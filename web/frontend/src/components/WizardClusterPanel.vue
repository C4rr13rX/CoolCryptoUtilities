<template>
  <details class="cluster-panel" @toggle="onToggle">
    <summary>Cluster and experience exchange</summary>
    <p role="status">{{ error || (busy ? 'Checking…' : '') }}</p>
    <template v-if="status">
      <p v-if="status.membership">Cluster members: {{ status.membership.members.length }}</p>
      <p v-else>{{ status.membership_error }}</p>
      <form class="actions" @submit.prevent="savePolicy">
        <label><input v-model="autoApprove" type="checkbox"> Auto-approve compatible pages</label>
        <label><input v-model="autoDeny" type="checkbox"> Auto-deny duplicate pages</label>
        <button :disabled="busy">Save policy</button>
      </form>
      <p>Requests with missing evidence, conflicting history, or unresolved references go to human review.</p>
      <form class="actions" @submit.prevent="requestPage">
        <label>
          Peer
          <select v-model="peer" required>
            <option disabled value="">Select a peer</option>
            <option v-for="name in status.peers" :key="name">{{ name }}</option>
          </select>
        </label>
        <label>Pool <input v-model.number="pool" type="number" min="0" max="4294967295" required></label>
        <button :disabled="busy || !peer">Request experience</button>
        <button type="button" :disabled="busy" @click="refresh">Refresh</button>
      </form>
      <ul v-if="status.requests.length">
        <li v-for="item in status.requests" :key="item.id">
          <strong>{{ item.peer }} · pool {{ item.pool }}</strong> · {{ item.state }}
          <p>{{ item.detail }}</p>
          <p v-if="item.evidence.peer_slots !== undefined">
            Local / peer slots: {{ item.evidence.local_slots }} / {{ item.evidence.peer_slots }} ·
            Page: {{ item.evidence.proposed_neurons }} neurons, {{ item.evidence.payload_bytes }} bytes ·
            Peer tick: {{ item.evidence.peer_tick }}
          </p>
          <div v-if="item.state === 'pending'" class="actions">
            <button v-if="item.evidence.decision === 'compatible'" :disabled="busy" @click="decide(item.id, 'approve')">Approve</button>
            <button :disabled="busy" @click="decide(item.id, 'deny')">Deny</button>
          </div>
        </li>
      </ul>
      <p v-else>No experience requests yet.</p>
    </template>
  </details>
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'

interface Policy { auto_approve: boolean; auto_deny: boolean }
interface Evidence {
  decision: string
  local_slots?: number
  peer_slots?: number
  proposed_neurons?: number
  payload_bytes?: number
  peer_tick?: number
}
interface ClusterStatus {
  peers: string[]
  policy: Policy
  membership?: { members: unknown[] }
  membership_error?: string
  requests: { id: string; peer: string; pool: number; state: string; detail: string; evidence: Evidence }[]
}
const props = defineProps<{ brainId: string }>()
const status = ref<ClusterStatus>()
const open = ref(false), busy = ref(false), error = ref('')
const autoApprove = ref(false), autoDeny = ref(false)
const peer = ref(''), pool = ref(1)
let controller: AbortController | undefined

function endpoint() {
  const search = new URLSearchParams({ brain_id: props.brainId })
  return '/api/wizard-chat/cluster/?' + search.toString()
}

function csrfToken() {
  const cookie = document.cookie.split(';').map(value => value.trim()).find(value => value.startsWith('csrftoken='))
  return cookie ? decodeURIComponent(cookie.slice('csrftoken='.length))
    : document.querySelector<HTMLInputElement>('input[name=csrfmiddlewaretoken]')?.value || ''
}

async function refresh() {
  controller?.abort()
  const operation = new AbortController()
  controller = operation
  busy.value = true
  error.value = ''
  try {
    const response = await fetch(endpoint(), { signal: operation.signal })
    const data = await response.json()
    if (!response.ok) throw new Error(data.error || 'Cluster status unavailable.')
    status.value = data
    autoApprove.value = data.policy.auto_approve
    autoDeny.value = data.policy.auto_deny
    if (!data.peers.includes(peer.value)) peer.value = ''
  } catch (issue) {
    if (!operation.signal.aborted) error.value = issue instanceof Error ? issue.message : String(issue)
  } finally {
    if (controller === operation) busy.value = false
  }
}

async function submit(data: object) {
  busy.value = true
  error.value = ''
  const target = endpoint()
  try {
    const response = await fetch(target, {
       method: 'POST',
       headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken() },
       body: JSON.stringify(data),
    })
    const result = await response.json()
    if (!response.ok) throw new Error(result.error || 'Cluster request failed.')
    if (target === endpoint()) await refresh()
  } catch (issue) {
    if (target === endpoint()) error.value = issue instanceof Error ? issue.message : String(issue)
  } finally {
    if (target === endpoint()) busy.value = false
  }
}
function savePolicy() { return submit({ action: 'policy', auto_approve: autoApprove.value, auto_deny: autoDeny.value }) }
function requestPage() { return submit({ action: 'request', peer: peer.value, pool: pool.value }) }
function decide(id: string, action: 'approve' | 'deny') { return submit({ id, action }) }
function onToggle(event: Event) {
  open.value = (event.target as HTMLDetailsElement).open
  if (open.value && !status.value) void refresh()
}
watch(() => props.brainId, () => {
  status.value = undefined
  peer.value = ''
  if (open.value) void refresh()
})
onBeforeUnmount(() => controller?.abort())
</script>

<style scoped>
.cluster-panel { flex-shrink: 0; padding: .5rem 1rem; max-height: 40vh; overflow: auto; color: #d9c6e5; font-size: .8rem; border-top: 1px solid #bda4cd30; }
summary { cursor: pointer; }
.actions { display: flex; align-items: center; flex-wrap: wrap; gap: .65rem; margin-block: .55rem; }
label { display: inline-flex; align-items: center; gap: .35rem; }
input[type=number] { width: 5.5rem; }
select, input[type=number], button { padding: .25rem .45rem; background: #170f24; color: inherit; border: 1px solid #75617e; border-radius: .4rem; }
button { cursor: pointer; }
button:disabled { opacity: .5; cursor: default; }
li { margin-bottom: .7rem; }
li p { margin: .2rem 0; }
</style>
