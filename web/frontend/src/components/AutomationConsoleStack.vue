<template>
  <div class="console-stack">
    <article class="console-card">
      <header>
        <span class="label">Production Manager Console</span>
      </header>
      <pre class="console-output" ref="managerEl">{{ managerText }}</pre>
    </article>
    <article class="console-card">
      <header>
        <span class="label">Guardian · Codex Console</span>
      </header>
      <pre class="console-output" ref="guardianEl">{{ guardianText }}</pre>
    </article>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue';

const props = defineProps<{
  managerLines?: string[];
  guardianLines?: string[];
}>();

const managerEl = ref<HTMLElement | null>(null);
const guardianEl = ref<HTMLElement | null>(null);

const managerText = computed(() => (props.managerLines?.length ? props.managerLines.join('\n') : 'No output yet.'));
const guardianText = computed(() =>
  props.guardianLines?.length ? props.guardianLines.join('\n') : 'No guardian transcript yet.'
);

function scrollToBottom(el: HTMLElement | null, force = false) {
  if (!el) return;
  const distanceFromBottom = el.scrollHeight - el.clientHeight - el.scrollTop;
  const nearBottom = distanceFromBottom <= 40;
  if (force || nearBottom) {
    el.scrollTop = el.scrollHeight;
  }
}

onMounted(() => {
  nextTick(() => {
    scrollToBottom(managerEl.value, true);
    scrollToBottom(guardianEl.value, true);
  });
});

watch(
  () => props.managerLines?.length || 0,
  () => nextTick(() => scrollToBottom(managerEl.value)),
);

watch(
  () => props.guardianLines?.length || 0,
  () => nextTick(() => scrollToBottom(guardianEl.value)),
);
</script>

<style scoped>
.console-stack {
  display: grid;
  gap: 1rem;
}

@media (min-width: 768px) {
  .console-stack {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.console-card {
  background: rgba(6, 12, 22, 0.9);
  border: 1px solid rgba(90, 141, 255, 0.25);
  border-radius: 16px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.console-card header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
}

.console-card .label {
  text-transform: uppercase;
  letter-spacing: 0.14rem;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.72);
}

.console-output {
  min-height: 220px;
  max-height: 360px;
  overflow-y: auto;
  background: rgba(0, 0, 0, 0.35);
  border-radius: 12px;
  padding: 0.8rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.82rem;
  line-height: 1.35;
  color: rgba(222, 239, 255, 0.9);
  border: 1px solid rgba(59, 130, 246, 0.22);
}
</style>
