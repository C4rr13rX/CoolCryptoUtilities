<template>
  <div class="status-light" :data-level="computedLevel">
    <span>{{ icon }}</span>
    <strong>{{ label }}</strong>
    <small v-if="detail" class="detail">{{ detail }}</small>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  level?: 'ok' | 'warn' | 'error';
  label: string;
  detail?: string;
}>();

const computedLevel = computed(() => props.level ?? 'ok');

const icon = computed(() => {
  switch (computedLevel.value) {
    case 'warn':
      return '🟠';
    case 'error':
      return '🔴';
    default:
      return '🟢';
  }
});
</script>

<style scoped>
.detail {
  color: rgba(255, 255, 255, 0.65);
  font-size: 0.75rem;
}
</style>
