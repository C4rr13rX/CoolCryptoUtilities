<template>
  <select class="model-select" :value="modelValue" @change="onChange">
    <option value="">{{ siteDefaultLabel }}</option>
    <option v-if="modelValue && !known.has(modelValue)" :value="modelValue">
      {{ modelValue }} · {{ t('models.current') }}
    </option>
    <optgroup v-if="premium.length" :label="t('models.premium')">
      <option v-for="m in premium" :key="m" :value="m">{{ m }}</option>
    </optgroup>
    <optgroup
      v-for="grp in catalogGroups"
      :key="grp.provider"
      :label="`${t('models.free_catalog')} · ${grp.provider}`"
    >
      <option v-for="m in grp.models" :key="m.id" :value="m.id">
        {{ m.configured ? '✓ ' : '' }}{{ m.id }}
      </option>
    </optgroup>
  </select>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { fetchModelOptions, type ModelOptions } from '@/api';
import { t } from '@/i18n';

const props = withDefaults(
  defineProps<{
    modelValue: string;
    // 'codex' -> OpenAI/codex premium set; 'c0d3r'/'all' -> claude+bedrock+openai
    kind?: 'codex' | 'c0d3r' | 'all';
    // Label for the empty option. Defaults to the site-default label; pages that
    // set the site default itself (Model Control) pass e.g. "Backend default".
    emptyLabel?: string;
  }>(),
  { kind: 'all', emptyLabel: '' }
);
const emit = defineEmits<{ (e: 'update:modelValue', value: string): void }>();

const options = ref<ModelOptions | null>(null);
onMounted(async () => {
  try {
    options.value = await fetchModelOptions();
  } catch {
    options.value = null;
  }
});

const siteDefaultLabel = computed(() => {
  if (props.emptyLabel) return props.emptyLabel;
  const label = options.value?.default?.label;
  return label ? `${t('models.site_default')} · ${label}` : t('models.site_default');
});

const premium = computed<string[]>(() => {
  const curated = options.value?.curated || {};
  const groups =
    props.kind === 'codex' ? ['codex', 'openai'] : ['claude', 'bedrock', 'openai'];
  const out: string[] = [];
  for (const g of groups) for (const m of curated[g] || []) if (!out.includes(m)) out.push(m);
  return out;
});

const catalogGroups = computed(() => {
  const cat = options.value?.catalog || [];
  const byProvider = new Map<string, Array<{ id: string; configured: boolean }>>();
  for (const m of cat) {
    if (!byProvider.has(m.provider)) byProvider.set(m.provider, []);
    byProvider.get(m.provider)!.push({ id: m.id, configured: m.configured });
  }
  return Array.from(byProvider.entries()).map(([provider, models]) => ({ provider, models }));
});

const known = computed<Set<string>>(() => {
  const s = new Set<string>(premium.value);
  for (const m of options.value?.catalog || []) s.add(m.id);
  return s;
});

const onChange = (e: Event) => {
  emit('update:modelValue', (e.target as HTMLSelectElement).value);
};
</script>

<style scoped>
.model-select {
  width: 100%;
}
</style>
