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
          <p>Paste entries (comments beginning with # are ignored). The parsed values populate the Default category.</p>
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
        <div class="actions">
          <button type="submit" class="btn" :disabled="!importForm.content.trim()">Import</button>
          <button type="button" class="btn ghost" @click="resetImport">Clear Text</button>
        </div>
      </form>
    </section>

    <section class="panel" v-if="showCreate">
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
    </section>

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
</template>

<script setup lang="ts">
import { reactive, ref, onMounted, computed } from 'vue';
import { useSecureSettingsStore } from '@/stores/secureSettings';

const store = useSecureSettingsStore();
const showCreate = ref(false);
const editing = ref<number | null>(null);
const confirmingClear = ref(false);
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
});

onMounted(() => {
  refresh();
});

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

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.6rem;
}
</style>
