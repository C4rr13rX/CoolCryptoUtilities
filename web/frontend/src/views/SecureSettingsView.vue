<template>
  <div class="secure-settings">
    <section class="panel">
      <header>
        <div>
          <h1>Secure Settings</h1>
          <p>Vaulted values pulled into every subprocess (Guardian, console, wallet, labs).</p>
        </div>
        <div class="header-actions">
          <button type="button" class="btn ghost" @click="refresh" :disabled="store.loading">
            {{ store.loading ? 'Refreshing…' : 'Refresh' }}
          </button>
          <button
            v-if="wizardSteps.length"
            type="button"
            class="btn"
            @click="wizardOpen = true"
            :disabled="store.loading"
          >
            Open Pipeline Wizard
          </button>
          <button type="button" class="btn danger" @click="confirmingClear = true" :disabled="store.loading || !store.items.length">
            Clear All
          </button>
        </div>
      </header>

      <div v-for="group in groupedSettings" :key="group.key" class="category-block">
        <h3 class="category-title">{{ group.label }}</h3>
        <div class="settings-grid">
          <article v-for="item in group.items" :key="item.name + String(item.id)" class="setting-card">
            <div class="card-head">
              <div>
                <strong>{{ item.label || item.name }}</strong>
                <small class="field-code">{{ item.name }}</small>
              </div>
              <div class="card-actions">
                <button type="button" class="link" @click="edit(item)">{{ item.is_placeholder ? 'Add' : 'Edit' }}</button>
                <button v-if="item.id" type="button" class="link danger" @click="remove(item)">Delete</button>
              </div>
            </div>
            <div class="value-row">
              <span v-if="item.is_secret">
                <template v-if="item.id && revealVisible[item.id]">
                  {{ revealValues[item.id] || '—' }}
                </template>
                <template v-else>
                  {{ item.preview || '••••••' }}
                </template>
              </span>
              <span v-else>{{ item.preview || 'Not set' }}</span>
              <button
                v-if="item.is_secret && item.id"
                type="button"
                class="link"
                @click="toggleReveal(item)"
              >
                {{ item.id && revealVisible[item.id] ? 'Hide' : 'Reveal' }}
              </button>
            </div>
            <p v-if="item.is_placeholder" class="placeholder-note">This value has not been configured yet.</p>
          </article>
        </div>
      </div>
      <article class="setting-card add-card" @click="showCreate = true">
        <span>+ Add Setting</span>
      </article>
    </section>

    <section class="panel">
      <header>
        <div>
          <h2>Import from .env</h2>
          <p>Paste entries or upload a .env file. Parsed values populate the Default category.</p>
        </div>
        <div class="header-actions">
          <button type="button" class="btn ghost" @click="downloadExport" :disabled="store.loading">
            Export .env
          </button>
        </div>
        <label class="switch-row import-secret-toggle">
          <input type="checkbox" v-model="importForm.is_secret" />
          <span>Store imported values as secrets</span>
        </label>
      </header>
      <form class="form-grid" @submit.prevent="importEnv">
        <label>
          <span>.env contents</span>
          <textarea v-model="importForm.content" rows="6" placeholder="API_KEY=example123
RPC_URL=https://..." />
        </label>
        <label>
          <span>.env file</span>
          <input type="file" accept=".env,text/plain" @change="handleFile" />
        </label>
        <div class="actions">
          <button type="submit" class="btn" :disabled="!importForm.content.trim()">Import</button>
          <button type="button" class="btn ghost" @click="importFile" :disabled="!importForm.file">Import File</button>
          <button type="button" class="btn ghost" @click="resetImport">Clear Text</button>
        </div>
      </form>
    </section>

    <div v-if="showCreate" class="modal-backdrop" @click.self="cancel">
      <div class="modal-card wide">
        <header>
          <div>
            <h2>{{ editing ? 'Edit Setting' : 'New Setting' }}</h2>
          </div>
        </header>
        <form class="form-grid" @submit.prevent="save">
          <label>
            <span>Name</span>
            <input type="text" v-model="form.name" required />
          </label>
          <label>
            <span>Category</span>
            <input type="text" v-model="form.category" placeholder="default" />
          </label>
          <label class="switch-row">
            <input type="checkbox" v-model="form.is_secret" />
            <span>Secret value</span>
          </label>
          <label>
            <span>Value</span>
            <textarea v-model="form.value" rows="4"></textarea>
          </label>
          <div class="actions">
            <button type="submit" class="btn">{{ editing ? 'Update' : 'Save' }}</button>
            <button type="button" class="btn ghost" @click="cancel">Cancel</button>
          </div>
        </form>
      </div>
    </div>

    <div v-if="confirmingClear" class="modal-backdrop">
      <div class="modal-card">
        <h3>Clear all settings?</h3>
        <p>This removes every stored value for your account. This cannot be undone.</p>
        <div class="modal-actions">
          <button type="button" class="btn danger" @click="clearAll">Yes</button>
          <button type="button" class="btn ghost" @click="confirmingClear = false">Cancel</button>
        </div>
      </div>
    </div>
  </div>
  <TradingStartupWizard
    v-if="wizardSteps.length"
    v-model:open="wizardOpen"
    :steps="wizardSteps"
    title="Pipeline Setup Wizard"
    subtitle="Add the missing keys so the pipeline can run end-to-end."
    eyebrow="Settings Wizard"
  >
    <template v-for="step in wizardSteps" #[`step-${step.id}`]="{ step: slotStep }">
      <div class="wizard-field">
        <label :for="`wizard-${step.id}`">{{ step.label }}</label>
        <textarea
          v-if="step.input === 'textarea'"
          :id="`wizard-${step.id}`"
          v-model="wizardValues[step.id]"
          rows="3"
          :placeholder="step.placeholder"
        ></textarea>
        <input
          v-else
          :id="`wizard-${step.id}`"
          :type="step.is_secret ? 'password' : 'text'"
          v-model="wizardValues[step.id]"
          :placeholder="step.placeholder"
        />
        <div class="wizard-actions">
          <button class="btn" type="button" @click="saveWizardStep(step)" :disabled="!wizardValues[step.id]?.trim() || wizardSaving[step.id]">
            {{ wizardSaving[step.id] ? 'Saving…' : 'Save' }}
          </button>
          <button class="btn ghost" type="button" @click="clearWizardStep(step)">
            Clear
          </button>
        </div>
        <p v-if="wizardErrors[step.id]" class="wizard-error">{{ wizardErrors[step.id] }}</p>
      </div>
    </template>
  </TradingStartupWizard>
</template>

<script setup lang="ts">
import { reactive, ref, onMounted, computed, watch } from 'vue';
import { useSecureSettingsStore } from '@/stores/secureSettings';
import TradingStartupWizard from '@/components/TradingStartupWizard.vue';

const store = useSecureSettingsStore();
const showCreate = ref(false);
const editing = ref<number | null>(null);
const confirmingClear = ref(false);
const wizardOpen = ref(true);
const wizardValues = reactive<Record<string, string>>({});
const wizardSaving = reactive<Record<string, boolean>>({});
const wizardErrors = reactive<Record<string, string>>({});
const revealVisible = reactive<Record<number, boolean>>({});
const revealValues = reactive<Record<number, string>>({});
const form = reactive({
  name: '',
  category: 'Default',
  is_secret: true,
  value: '',
});
const importForm = reactive({
  content: '',
  is_secret: true,
  file: null as File | null,
});

onMounted(() => {
  refresh();
});

type WizardStep = {
  id: string;
  name: string;
  label: string;
  title: string;
  description: string;
  detail?: string;
  placeholder?: string;
  is_secret: boolean;
  input?: 'text' | 'textarea';
  tone?: 'info' | 'warning' | 'critical' | 'success';
};

const pipelineRequirements: WizardStep[] = [
  {
    id: 'mnemonic',
    name: 'MNEMONIC',
    label: 'Wallet recovery phrase',
    title: 'Add wallet recovery phrase',
    description: 'Needed to sign transactions and run live trading.',
    detail: 'Stored securely in the vault.',
    placeholder: 'problem tube idea ...',
    is_secret: true,
    input: 'textarea',
    tone: 'critical',
  },
  {
    id: 'alchemy',
    name: 'ALCHEMY_API_KEY',
    label: 'Alchemy API Key',
    title: 'Connect RPC provider',
    description: 'RPC + NFT data for Base/ETH monitoring.',
    detail: 'Used for wallet sync, price sampling, and on-chain reads.',
    placeholder: 'Alchemy API key',
    is_secret: true,
    tone: 'critical',
  },
  {
    id: 'cryptopanic',
    name: 'CRYPTOPANIC_API_KEY',
    label: 'CryptoPanic API Key',
    title: 'Enable news ingest',
    description: 'Unlocks headline + news sentiment feeds.',
    placeholder: 'CryptoPanic API key',
    is_secret: true,
    tone: 'warning',
  },
  {
    id: 'thegraph',
    name: 'THEGRAPH_API_KEY',
    label: 'The Graph API Key',
    title: 'Enable subgraph data',
    description: 'Needed for Uniswap/DEX data and analytics.',
    placeholder: 'The Graph API key',
    is_secret: true,
    tone: 'warning',
  },
  {
    id: 'zerox',
    name: 'ZEROX_API_KEY',
    label: '0x API Key',
    title: 'Enable swap routing',
    description: 'Unlocks swap pricing + routing quotes.',
    placeholder: '0x API key',
    is_secret: true,
    tone: 'warning',
  },
  {
    id: 'lifi',
    name: 'LIFI_API_KEY',
    label: 'LI.FI API Key',
    title: 'Enable bridging',
    description: 'Bridge + swap orchestration for cross-chain moves.',
    placeholder: 'LI.FI API key',
    is_secret: true,
    tone: 'warning',
  },
  {
    id: 'ankr',
    name: 'ANKR_API_KEY',
    label: 'Ankr API Key',
    title: 'Enable historical data pulls',
    description: 'Provides multichain RPC + historical queries.',
    placeholder: 'Ankr API key',
    is_secret: true,
    tone: 'info',
  },
  {
    id: 'goplus-key',
    name: 'GOPLUS_APP_KEY',
    label: 'GoPlus App Key',
    title: 'Enable token safety checks',
    description: 'Scam filtering uses GoPlus for security scoring.',
    placeholder: 'GoPlus app key',
    is_secret: true,
    tone: 'warning',
  },
  {
    id: 'goplus-secret',
    name: 'GOPLUS_APP_SECRET',
    label: 'GoPlus App Secret',
    title: 'Confirm GoPlus secret',
    description: 'Pairs with the GoPlus app key.',
    placeholder: 'GoPlus app secret',
    is_secret: true,
    tone: 'warning',
  },
];

const groupedSettings = computed(() => {
  const groups: Record<string, any[]> = {};
  store.items.forEach((item: any) => {
    const category = (item.category || 'default').toString();
    const key = category.toLowerCase();
    if (!groups[key]) {
      groups[key] = [];
    }
    groups[key].push(item);
  });
  return Object.entries(groups).map(([key, items]) => ({
    key,
    label: key === 'default' ? 'Default' : items[0]?.category || key,
    items,
  }));
});

const settingsByName = computed(() => {
  const map = new Map<string, any>();
  store.items.forEach((item: any) => {
    if (item?.name) {
      map.set(item.name, item);
    }
  });
  return map;
});

const wizardSteps = computed<WizardStep[]>(() => {
  const steps: WizardStep[] = [];
  pipelineRequirements.forEach((req) => {
    const entry = settingsByName.value.get(req.name);
    const missing = !entry || entry.is_placeholder;
    if (missing) {
      steps.push(req);
    }
  });
  return steps;
});

watch(
  () => wizardSteps.value,
  (steps) => {
    steps.forEach((step) => {
      if (!(step.id in wizardValues)) {
        wizardValues[step.id] = '';
      }
    });
    if (!steps.length) {
      wizardOpen.value = false;
    }
  },
  { immediate: true }
);

function refresh() {
  store.load();
}

async function clearAll() {
  confirmingClear.value = false;
  await store.clearAll();
}

function edit(item: any) {
  editing.value = item.id || null;
  form.name = item.name;
  form.category = item.category ? (item.category === 'default' ? 'Default' : item.category) : 'Default';
  form.is_secret = !!item.is_secret;
  form.value = '';
  showCreate.value = true;
}

function remove(item: any) {
  if (!item.id) return;
  if (!confirm(`Delete ${item.name}?`)) return;
  store.remove(item.id);
}

function cancel() {
  showCreate.value = false;
  editing.value = null;
  form.name = '';
  form.category = 'Default';
  form.is_secret = true;
  form.value = '';
}

async function save() {
  const categoryInput = (form.category || 'Default').trim() || 'Default';
  const storedCategory = categoryInput.toLowerCase() === 'default' ? 'default' : categoryInput;
  await store.save({
    id: editing.value || undefined,
    name: form.name,
    category: storedCategory,
    is_secret: form.is_secret,
    value: form.value,
  });
  cancel();
}

async function importEnv() {
  if (!importForm.content.trim()) return;
  await store.importFromEnv({
    content: importForm.content,
    category: 'default',
    is_secret: importForm.is_secret,
  });
  resetImport();
}

function resetImport() {
  importForm.content = '';
  importForm.file = null;
}

async function toggleReveal(item: any) {
  if (!item.id) return;
  if (revealVisible[item.id]) {
    revealVisible[item.id] = false;
    delete revealValues[item.id];
    return;
  }
  const data = await store.reveal(item.id);
  const secret = data?.revealed_value || data?.value || '';
  revealValues[item.id] = secret;
  revealVisible[item.id] = true;
}

function handleFile(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target?.files?.[0] || null;
  importForm.file = file;
}

async function importFile() {
  if (!importForm.file) return;
  await store.importFromEnvFile(importForm.file, importForm.is_secret);
  importForm.file = null;
}

async function downloadExport() {
  const content = await store.exportEnv();
  const blob = new Blob([content || ''], { type: 'text/plain;charset=utf-8' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'secure_settings.env';
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

async function saveWizardStep(step: WizardStep) {
  const value = (wizardValues[step.id] || '').trim();
  if (!value) return;
  wizardSaving[step.id] = true;
  wizardErrors[step.id] = '';
  try {
    await store.save({
      name: step.name,
      category: 'default',
      is_secret: step.is_secret,
      value,
    });
    wizardValues[step.id] = '';
  } catch (error: any) {
    wizardErrors[step.id] = error?.message || 'Failed to save setting';
  } finally {
    wizardSaving[step.id] = false;
  }
}

function clearWizardStep(step: WizardStep) {
  wizardValues[step.id] = '';
  wizardErrors[step.id] = '';
}
</script>

<style scoped>
.secure-settings {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.header-actions {
  display: flex;
  gap: 0.6rem;
  align-items: center;
}

.panel {
  background: rgba(9, 15, 24, 0.85);
  border-radius: 24px;
  padding: 1.4rem 1.6rem;
  border: 1px solid rgba(59, 130, 246, 0.18);
}

.category-block + .category-block {
  margin-top: 1.2rem;
}

.category-title {
  margin: 0 0 0.6rem;
  font-size: 0.95rem;
  letter-spacing: 0.2rem;
  color: #cbd5f5;
  text-transform: uppercase;
}

.settings-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1rem;
}

.setting-card {
  border: 1px solid rgba(59, 130, 246, 0.2);
  border-radius: 16px;
  padding: 1rem;
  background: rgba(10, 19, 34, 0.85);
}

.setting-card.add-card {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #93c5fd;
  cursor: pointer;
  font-weight: 600;
  border-style: dashed;
}

.card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-actions {
  display: flex;
  gap: 0.5rem;
}

.field-code {
  display: block;
  font-size: 0.75rem;
  letter-spacing: 0.08rem;
  color: rgba(255, 255, 255, 0.5);
}

.value-row {
  margin-top: 0.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
}

.placeholder-note {
  margin: 0.4rem 0 0;
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.55);
}

.link {
  background: none;
  border: none;
  color: #93c5fd;
  cursor: pointer;
  padding: 0;
}

.form-grid {
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}

.form-grid input,
.form-grid textarea {
  border-radius: 12px;
  border: 1px solid rgba(59, 130, 246, 0.3);
  background: rgba(7, 14, 25, 0.85);
  color: #e2e8f0;
  padding: 0.6rem;
}

.switch-row {
  flex-direction: row;
  align-items: center;
  gap: 0.5rem;
}

.actions {
  display: flex;
  gap: 0.8rem;
}

.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(1, 3, 8, 0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 40;
}

.modal-card {
  background: rgba(9, 15, 24, 0.95);
  border-radius: 20px;
  padding: 1.5rem;
  width: min(90vw, 360px);
  border: 1px solid rgba(248, 113, 113, 0.4);
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.modal-card.wide {
  width: min(92vw, 520px);
  border-color: rgba(59, 130, 246, 0.35);
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.6rem;
}

.wizard-field {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.wizard-field label {
  font-weight: 600;
  color: #dbeafe;
}

.wizard-field textarea,
.wizard-field input {
  background: rgba(4, 10, 20, 0.75);
  border: 1px solid rgba(96, 165, 250, 0.35);
  border-radius: 12px;
  padding: 0.6rem 0.8rem;
  color: #e2e8f0;
}

.wizard-actions {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.wizard-error {
  margin: 0;
  color: #fca5a5;
  font-size: 0.85rem;
}
</style>
