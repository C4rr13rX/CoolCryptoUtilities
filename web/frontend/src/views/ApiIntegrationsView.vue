<template>
  <div class="integrations-view">
    <section class="panel">
      <header>
        <div>
          <h2>API Integrations</h2>
          <p class="caption">Provider credentials stored in the secure vault.</p>
        </div>
        <button class="btn ghost" type="button" @click="store.load">Refresh</button>
      </header>
      <div class="integration-grid">
        <article v-for="item in store.items" :key="item.name" class="integration-card">
          <header>
            <div>
              <strong>{{ item.label }}</strong>
              <span class="caption">{{ item.description }}</span>
            </div>
            <a v-if="item.url" class="link" :href="item.url" target="_blank" rel="noreferrer">Manage</a>
          </header>
          <div class="stored-value">
            <span>{{ displayValue(item) }}</span>
            <button
              v-if="item.has_value"
              class="link"
              type="button"
              @click="toggleReveal(item.name)"
            >
              {{ store.revealVisible[item.name] ? 'Hide' : 'Reveal' }}
            </button>
          </div>
          <label>
            <span>New value</span>
            <input type="text" v-model="formState[item.name]" placeholder="Enter key" />
          </label>
          <div class="actions">
            <button class="btn" type="button" @click="save(item)">Save</button>
            <button class="btn ghost" type="button" @click="clear(item)">Clear</button>
            <button
              v-if="item.can_test"
              class="btn ghost"
              type="button"
              :disabled="store.testing[item.name]"
              @click="test(item)"
            >
              {{ store.testing[item.name] ? 'Testing…' : 'Test' }}
            </button>
          </div>
          <p v-if="store.testResult[item.name]" class="test-result">{{ store.testResult[item.name] }}</p>
        </article>
      </div>
      <p v-if="!store.items.length" class="empty-text">No integrations configured.</p>
    </section>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, watch } from 'vue';
import { useIntegrationsStore } from '@/stores/integrations';

const store = useIntegrationsStore();
const formState = reactive<Record<string, string>>({});

onMounted(() => {
  store.load();
});

watch(
  () => store.items,
  (items) => {
    items.forEach((item: any) => {
      if (formState[item.name] === undefined) {
        formState[item.name] = '';
      }
    });
  },
  { immediate: true }
);

watch(
  () => store.revealState,
  (state) => {
    Object.entries(state).forEach(([name, value]) => {
      if (value) {
        formState[name] = value as string;
      }
    });
  },
  { deep: true }
);

function displayValue(item: any) {
  if (store.revealVisible[item.name]) {
    return store.revealState[item.name] || '—';
  }
  return item.has_value ? '••••••' : 'Not set';
}

function toggleReveal(name: string) {
  store.reveal(name);
}

async function save(item: any) {
  await store.save(item.name, formState[item.name] || null);
  formState[item.name] = '';
}

async function clear(item: any) {
  formState[item.name] = '';
  await store.save(item.name, null);
}

async function test(item: any) {
  const value = formState[item.name] || '';
  try {
    await store.test(item.name, value);
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn('Integration test failed', error);
  }
}
</script>

<style scoped>
.integrations-view {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.panel {
  background: rgba(11, 22, 37, 0.85);
  border: 1px solid rgba(111, 167, 255, 0.18);
  border-radius: 18px;
  padding: 1.3rem 1.5rem;
}

.panel header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
}

.integration-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1rem;
}

.integration-card {
  border: 1px solid rgba(111, 167, 255, 0.18);
  border-radius: 16px;
  padding: 1rem;
  background: rgba(7, 14, 25, 0.85);
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}

.integration-card header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 0.5rem;
}

.integration-card label {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.integration-card input {
  border-radius: 10px;
  border: 1px solid rgba(111, 167, 255, 0.25);
  background: rgba(255, 255, 255, 0.03);
  color: #fefefe;
  padding: 0.45rem 0.7rem;
}

.stored-value {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.85);
}

.actions {
  display: flex;
  gap: 0.5rem;
}

.test-result {
  margin: 0;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.7);
}

.empty-text {
  color: rgba(255, 255, 255, 0.6);
}
</style>
