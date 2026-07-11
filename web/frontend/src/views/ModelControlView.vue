<template>
  <div class="model-control-view">
    <section class="panel hero-panel">
      <header>
        <div>
          <p class="eyebrow">C0d3rV2 runtime</p>
          <h1>AI Model Control</h1>
          <p>Choose C0d3rV2's backend, constrain AgentTheFreeloader, and securely connect providers.</p>
        </div>
        <button type="button" class="btn ghost" :disabled="loading" @click="load">
          {{ loading ? 'Refreshing…' : 'Refresh' }}
        </button>
      </header>
      <div v-if="error" class="notice error">{{ error }}</div>
      <div v-if="notice" class="notice success">{{ notice }}</div>
    </section>

    <section class="panel">
      <header>
        <div>
          <h2>Active C0d3rV2 backend</h2>
          <p>Saving this selection resets cached C0d3rV2 sessions so the next request uses it.</p>
        </div>
        <span class="status-pill">{{ selectedBackendLabel }}</span>
      </header>

      <div class="backend-grid">
        <button
          v-for="backend in data?.backends || []"
          :key="backend.id"
          type="button"
          :class="['backend-card', { selected: form.backend === backend.id }]"
          @click="form.backend = backend.id"
        >
          <strong>{{ backend.label }}</strong>
          <span>{{ backend.description }}</span>
        </button>
      </div>

      <label v-if="form.backend === 'bedrock' || form.backend === 'openai' || form.backend === 'claude'" class="model-field">
        <span>Model ID override <small>(optional)</small></span>
        <input v-model.trim="form.model" type="text" placeholder="Leave blank for the backend default" />
      </label>

      <div class="actions">
        <button type="button" class="btn" :disabled="saving || loading" @click="saveConfig">
          {{ saving ? 'Applying…' : 'Apply backend selection' }}
        </button>
      </div>
    </section>

    <section class="panel">
      <header>
        <div>
          <h2>ATF correction telemetry</h2>
          <p>Validation failures and corrective retries attributed to the model that produced the failed step.</p>
        </div>
        <span class="status-pill">{{ data?.corrections?.length || 0 }} recent events</span>
      </header>
      <div class="correction-list">
        <article v-for="event in data?.corrections || []" :key="event.id" class="correction-row">
          <div class="correction-head">
            <strong>{{ event.provider }} / {{ event.model }}</strong>
            <span :class="['connection-dot', { hallucination: event.is_hallucination, online: event.resolved }]">
              {{ event.is_hallucination ? 'hallucination' : event.classification }} · {{ event.resolved ? 'corrected' : 'unresolved' }}
            </span>
          </div>
          <p>{{ event.trigger }}</p>
          <small>{{ new Date(event.created_at * 1000).toLocaleString() }} · {{ event.session || 'unnamed session' }}</small>
        </article>
        <div v-if="!data?.corrections?.length" class="empty">No correction events recorded yet.</div>
      </div>
    </section>

    <section v-if="form.backend === 'freeloader'" class="panel">
      <header>
        <div>
          <h2>AgentTheFreeloader model pool</h2>
          <p>Leave every model unchecked for automatic quality-and-quota routing, or restrict ATF to specific models.</p>
        </div>
        <button type="button" class="btn ghost" @click="form.atf_models = []">Use automatic routing</button>
      </header>

      <div class="provider-list">
        <article v-for="provider in data?.providers || []" :key="provider.name" class="provider-card">
          <div class="provider-head">
            <div>
              <strong>{{ provider.name }}</strong>
              <span :class="['connection-dot', { online: provider.configured }]">
                {{ provider.configured ? 'available' : 'credential needed' }}
              </span>
            </div>
            <button type="button" class="link" :disabled="!provider.configured" @click="toggleProvider(provider)">
              Toggle available
            </button>
          </div>
          <div class="model-list">
            <label v-for="model in provider.models" :key="provider.name + model.id" :class="['model-row', { unavailable: !model.configured }]">
              <input v-model="form.atf_models" type="checkbox" :value="model.id" :disabled="!model.configured" />
              <span>
                <strong>{{ model.id }}</strong>
                <small>{{ model.best_at || 'General model' }}</small>
              </span>
            </label>
          </div>
        </article>
      </div>
      <div class="actions sticky-actions">
        <span>{{ form.atf_models.length ? `${form.atf_models.length} constrained model(s)` : 'Automatic model routing' }}</span>
        <button type="button" class="btn" :disabled="saving" @click="saveConfig">Save ATF model pool</button>
      </div>
    </section>

    <section class="panel">
      <header>
        <div>
          <h2>Connections and credentials</h2>
          <p>Values are stored in the encrypted SecureSetting vault. Existing secrets are never returned to this page.</p>
        </div>
        <span class="status-pill">{{ configuredCredentialCount }} connected</span>
      </header>

      <div class="credential-grid">
        <article v-for="credential in data?.credentials || []" :key="credential.name" class="credential-card">
          <div class="credential-title">
            <div>
              <strong>{{ credential.label }}</strong>
              <code>{{ credential.name }}</code>
            </div>
            <span :class="['connection-dot', { online: credential.configured }]">
              {{ credential.configured ? credential.source : 'not set' }}
            </span>
          </div>
          <p>{{ credential.description }}</p>
          <div class="credential-entry">
            <input
              v-model="credentialDrafts[credential.name]"
              :type="credential.is_secret && !visible[credential.name] ? 'password' : 'text'"
              :placeholder="credential.configured ? 'Enter a replacement value' : 'Enter value'"
              autocomplete="off"
              @keyup.enter="saveCredential(credential)"
            />
            <button v-if="credential.is_secret" type="button" class="link" @click="visible[credential.name] = !visible[credential.name]">
              {{ visible[credential.name] ? 'Hide' : 'Show' }}
            </button>
          </div>
          <div class="card-actions">
            <button type="button" class="btn small" :disabled="busyCredential === credential.name || !credentialDrafts[credential.name]?.trim()" @click="saveCredential(credential)">
              Save
            </button>
            <button v-if="credential.source === 'vault'" type="button" class="btn ghost small" :disabled="busyCredential === credential.name" @click="removeCredential(credential)">
              Remove vault value
            </button>
          </div>
        </article>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue';
import {
  deleteModelCredential,
  fetchModelControl,
  saveModelControl,
  saveModelCredential,
  type ModelControlCredential,
  type ModelControlPayload,
} from '@/api';

const data = ref<ModelControlPayload | null>(null);
const loading = ref(false);
const saving = ref(false);
const error = ref('');
const notice = ref('');
const busyCredential = ref('');
const credentialDrafts = reactive<Record<string, string>>({});
const visible = reactive<Record<string, boolean>>({});
const form = reactive({ backend: 'wizard', model: '', atf_models: [] as string[] });

const selectedBackendLabel = computed(() => data.value?.backends.find(item => item.id === form.backend)?.label || form.backend);
const configuredCredentialCount = computed(() => data.value?.credentials.filter(item => item.configured).length || 0);

const showMessage = (message: string) => {
  notice.value = message;
  window.setTimeout(() => { if (notice.value === message) notice.value = ''; }, 4000);
};

const load = async () => {
  loading.value = true;
  error.value = '';
  try {
    data.value = await fetchModelControl();
    form.backend = data.value.config.backend || 'wizard';
    form.model = data.value.config.model || '';
    form.atf_models = [...(data.value.config.atf_models || [])];
  } catch (err: any) {
    error.value = err?.response?.data?.detail || err?.message || 'Unable to load model control.';
  } finally {
    loading.value = false;
  }
};

const saveConfig = async () => {
  saving.value = true;
  error.value = '';
  try {
    await saveModelControl({ ...form, atf_models: [...form.atf_models] });
    showMessage('C0d3rV2 model selection saved. New requests will use this configuration.');
    await load();
  } catch (err: any) {
    error.value = err?.response?.data?.detail || err?.message || 'Unable to save model selection.';
  } finally {
    saving.value = false;
  }
};

const toggleProvider = (provider: NonNullable<ModelControlPayload['providers']>[number]) => {
  const ids = provider.models.filter(item => item.configured).map(item => item.id);
  const allSelected = ids.length > 0 && ids.every(id => form.atf_models.includes(id));
  if (allSelected) form.atf_models = form.atf_models.filter(id => !ids.includes(id));
  else form.atf_models = Array.from(new Set([...form.atf_models, ...ids]));
};

const saveCredential = async (credential: ModelControlCredential) => {
  const value = credentialDrafts[credential.name]?.trim();
  if (!value) return;
  busyCredential.value = credential.name;
  error.value = '';
  try {
    await saveModelCredential(credential.name, value);
    credentialDrafts[credential.name] = '';
    showMessage(`${credential.label} saved to the encrypted vault.`);
    await load();
  } catch (err: any) {
    error.value = err?.response?.data?.detail || err?.message || `Unable to save ${credential.label}.`;
  } finally {
    busyCredential.value = '';
  }
};

const removeCredential = async (credential: ModelControlCredential) => {
  busyCredential.value = credential.name;
  error.value = '';
  try {
    await deleteModelCredential(credential.name);
    showMessage(`${credential.label} removed from the vault.`);
    await load();
  } catch (err: any) {
    error.value = err?.response?.data?.detail || err?.message || `Unable to remove ${credential.label}.`;
  } finally {
    busyCredential.value = '';
  }
};

onMounted(load);
</script>

<style scoped>
.model-control-view { display: grid; gap: 1rem; }
.hero-panel { background: radial-gradient(circle at top right, rgba(45,117,196,.24), transparent 42%), var(--panel-bg, #0d1520); }
header { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; }
header h1, header h2 { margin: 0 0 .35rem; }
header p { margin: 0; color: var(--muted, #91a2b7); }
.eyebrow { color: #7fb0ff; font-size: .72rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
.backend-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: .75rem; margin-top: 1rem; }
.backend-card { min-height: 108px; padding: 1rem; text-align: left; color: inherit; border: 1px solid rgba(148,163,184,.18); border-radius: 12px; background: rgba(8,15,24,.72); cursor: pointer; }
.backend-card strong, .backend-card span { display: block; }
.backend-card span { margin-top: .4rem; color: #91a2b7; font-size: .82rem; line-height: 1.35; }
.backend-card:hover { border-color: rgba(127,176,255,.55); }
.backend-card.selected { border-color: #4d94e8; box-shadow: inset 0 0 0 1px #4d94e8; background: rgba(45,117,196,.16); }
.model-field { display: grid; gap: .45rem; margin-top: 1rem; max-width: 640px; }
.model-field input, .credential-entry input { width: 100%; min-width: 0; padding: .68rem .75rem; color: #e9f1fb; background: #080f18; border: 1px solid rgba(148,163,184,.25); border-radius: 8px; }
.actions { display: flex; align-items: center; justify-content: flex-end; gap: .75rem; margin-top: 1rem; }
.status-pill, .connection-dot { border-radius: 99px; padding: .28rem .6rem; font-size: .72rem; background: rgba(148,163,184,.13); color: #aab8ca; white-space: nowrap; }
.connection-dot.online { background: rgba(52,211,153,.13); color: #55e3af; }
.provider-list { display: grid; gap: .8rem; margin-top: 1rem; }
.provider-card, .credential-card { border: 1px solid rgba(148,163,184,.16); border-radius: 12px; background: rgba(8,15,24,.62); }
.provider-head { display: flex; justify-content: space-between; align-items: center; gap: .8rem; padding: .8rem 1rem; border-bottom: 1px solid rgba(148,163,184,.12); }
.provider-head strong { margin-right: .6rem; }
.model-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); padding: .35rem; }
.model-row { display: flex; gap: .65rem; align-items: flex-start; padding: .65rem; border-radius: 8px; }
.model-row:hover { background: rgba(127,176,255,.06); }
.model-row span, .model-row strong, .model-row small { display: block; min-width: 0; }
.model-row strong { overflow-wrap: anywhere; }
.model-row small { margin-top: .18rem; color: #91a2b7; }
.model-row.unavailable { opacity: .48; }
.sticky-actions { position: sticky; bottom: .5rem; padding: .75rem; border: 1px solid rgba(127,176,255,.22); border-radius: 10px; background: rgba(8,15,24,.96); }
.credential-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: .8rem; margin-top: 1rem; }
.credential-card { padding: 1rem; }
.credential-title { display: flex; justify-content: space-between; gap: .75rem; align-items: flex-start; }
.credential-title strong, .credential-title code { display: block; }
.credential-title code { margin-top: .25rem; color: #7fb0ff; font-size: .72rem; overflow-wrap: anywhere; }
.credential-card > p { min-height: 2.4em; color: #91a2b7; font-size: .82rem; }
.credential-entry { display: flex; align-items: center; gap: .55rem; }
.card-actions { display: flex; gap: .55rem; margin-top: .7rem; }
.btn.small { padding: .45rem .72rem; font-size: .78rem; }
.link { color: #7fb0ff; border: 0; background: transparent; cursor: pointer; }
.link:disabled { opacity: .4; cursor: default; }
.notice { padding: .75rem 1rem; margin-top: 1rem; border-radius: 9px; }
.notice.error { color: #ff9c9f; background: rgba(255,90,95,.12); }
.notice.success { color: #55e3af; background: rgba(52,211,153,.1); }
.correction-list { display: grid; gap: .65rem; margin-top: 1rem; }
.correction-row { padding: .8rem 1rem; border: 1px solid rgba(148,163,184,.16); border-radius: 10px; background: rgba(8,15,24,.62); }
.correction-head { display: flex; justify-content: space-between; align-items: flex-start; gap: .75rem; }
.correction-row p { margin: .55rem 0; color: #c0ccda; overflow-wrap: anywhere; }
.correction-row small { color: #718096; }
.connection-dot.hallucination { background: rgba(255,90,95,.12); color: #ff9c9f; }
@media (max-width: 700px) { header { flex-direction: column; } .credential-grid, .model-list { grid-template-columns: 1fr; } }
</style>
