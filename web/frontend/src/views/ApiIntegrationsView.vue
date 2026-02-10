<template>
  <div class="integrations-view">
    <section class="panel">
      <header>
        <div>
          <h2>{{ t('integrations.title') }}</h2>
          <p class="caption">{{ t('integrations.subtitle') }}</p>
        </div>
        <button class="btn ghost" type="button" @click="store.load">{{ t('common.refresh') }}</button>
      </header>
      <div class="integration-grid">
        <article v-for="item in store.items" :key="item.name" class="integration-card">
          <header>
            <div>
              <strong>{{ item.label }}</strong>
              <span class="caption">{{ item.description }}</span>
            </div>
            <a v-if="item.url" class="link" :href="item.url" target="_blank" rel="noreferrer">{{ t('integrations.manage') }}</a>
          </header>
          <div class="stored-value">
            <span>{{ displayValue(item) }}</span>
            <button
              v-if="item.has_value"
              class="link"
              type="button"
              @click="toggleReveal(item.name)"
            >
              {{ store.revealVisible[item.name] ? t('common.hide') : t('common.reveal') }}
            </button>
          </div>
          <div class="actions">
            <button class="btn" type="button" @click="openEditor(item)">
              {{ item.has_value ? t('common.update') : t('common.set') }}
            </button>
            <button class="btn ghost" type="button" @click="clearDirect(item)" :disabled="!item.has_value">
              {{ t('common.clear') }}
            </button>
          </div>
          <p v-if="store.testResult[item.name]" class="test-result">{{ store.testResult[item.name] }}</p>
        </article>
      </div>
      <p v-if="!store.items.length" class="empty-text">{{ t('integrations.empty') }}</p>
    </section>

    <div v-if="activeItem" class="modal-backdrop" @click.self="closeEditor">
      <div class="modal-card">
        <header>
          <div>
            <h3>{{ activeItem.label }}</h3>
            <p class="caption">{{ activeItem.description }}</p>
          </div>
        </header>
        <form class="form-grid" @submit.prevent="saveActive">
          <label>
            <span>{{ t('integrations.new_value') }}</span>
            <input type="text" v-model="draftValue" :placeholder="t('integrations.enter_key')" />
          </label>
          <div class="actions">
            <button class="btn" type="submit">{{ t('common.save') }}</button>
            <button class="btn ghost" type="button" @click="closeEditor">{{ t('common.cancel') }}</button>
            <button class="btn ghost" type="button" @click="clearActive" :disabled="!activeItem.has_value">
              {{ t('common.clear') }}
            </button>
            <button
              v-if="activeItem.can_test"
              class="btn ghost"
              type="button"
              :disabled="store.testing[activeItem.name]"
              @click="testActive"
            >
              {{ store.testing[activeItem.name] ? t('common.testing') : t('common.test') }}
            </button>
          </div>
        </form>
        <p v-if="store.testResult[activeItem.name]" class="test-result">{{ store.testResult[activeItem.name] }}</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { useIntegrationsStore } from '@/stores/integrations';
import { t } from '@/i18n';

const store = useIntegrationsStore();
const activeItem = ref<any | null>(null);
const draftValue = ref('');

onMounted(() => {
  store.load();
});

function displayValue(item: any) {
  if (store.revealVisible[item.name]) {
    return store.revealState[item.name] || t('common.none');
  }
  return item.has_value ? t('common.masked') : t('common.not_set');
}

function toggleReveal(name: string) {
  store.reveal(name);
}

function openEditor(item: any) {
  activeItem.value = item;
  draftValue.value = store.revealState[item.name] || '';
}

function closeEditor() {
  activeItem.value = null;
  draftValue.value = '';
}

async function saveActive() {
  if (!activeItem.value) return;
  await store.save(activeItem.value.name, draftValue.value || null);
  closeEditor();
}

async function clearActive() {
  if (!activeItem.value) return;
  await store.save(activeItem.value.name, null);
  closeEditor();
}

async function clearDirect(item: any) {
  await store.save(item.name, null);
}

async function testActive() {
  if (!activeItem.value) return;
  const value = draftValue.value || '';
  try {
    await store.test(activeItem.value.name, value);
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
  border-radius: 18px;
  padding: 1.4rem;
  width: min(92vw, 460px);
  border: 1px solid rgba(111, 167, 255, 0.35);
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}

.modal-card input {
  border-radius: 10px;
  border: 1px solid rgba(111, 167, 255, 0.25);
  background: rgba(255, 255, 255, 0.03);
  color: #fefefe;
  padding: 0.45rem 0.7rem;
}

.form-grid {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}
</style>
