<template>
  <div class="secure-settings">
    <section class="panel">
      <header>
        <div>
          <h1>Secure Settings</h1>
          <p>Store wallet keys, API credentials, and endpoint overrides per user.</p>
        </div>
        <button type="button" class="btn ghost" @click="refresh" :disabled="store.loading">
          {{ store.loading ? 'Refreshing…' : 'Refresh' }}
        </button>
      </header>

      <div v-for="group in groupedSettings" :key="group.key" class="category-block">
        <h3 class="category-title">{{ group.label }}</h3>
        <div class="settings-grid">
          <article v-for="item in group.items" :key="item.id" class="setting-card">
            <div class="card-head">
              <strong>{{ item.name }}</strong>
              <div class="card-actions">
                <button type="button" class="link" @click="edit(item)">Edit</button>
                <button type="button" class="link danger" @click="remove(item)">Delete</button>
              </div>
            </div>
            <p v-if="item.is_secret">Secret · {{ item.preview }}</p>
            <p v-else>{{ item.preview }}</p>
          </article>
        </div>
      </div>
      <article class="setting-card add-card" @click="showCreate = true">
        <span>+ Add Setting</span>
      </article>
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
  </div>
</template>

<script setup lang="ts">
import { reactive, ref, onMounted, computed } from 'vue';
import { useSecureSettingsStore } from '@/stores/secureSettings';

const store = useSecureSettingsStore();
const showCreate = ref(false);
const editing = ref<number | null>(null);
const form = reactive({
  name: '',
  category: 'Default',
  is_secret: true,
  value: '',
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

function edit(item: any) {
  editing.value = item.id;
  form.name = item.name;
  form.category = item.category ? (item.category === 'default' ? 'Default' : item.category) : 'Default';
  form.is_secret = !!item.is_secret;
  form.value = '';
  showCreate.value = true;
}

function remove(item: any) {
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
</script>

<style scoped>
.secure-settings {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
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

.link {
  background: none;
  border: none;
  color: #93c5fd;
  cursor: pointer;
}

.link.danger {
  color: #f87171;
}

.form-grid {
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}

.form-grid label {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
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
</style>
