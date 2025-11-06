<template>
  <div class="status-light" :data-level="computedLevel">
    <div class="glyph">
      <HackerIcon :name="computedIcon" :size="18" />
    </div>
    <div class="copy">
      <strong>{{ label }}</strong>
      <ul v-if="detailLines.length" class="detail-list">
        <li v-for="(line, idx) in detailLines" :key="line + idx">{{ line }}</li>
      </ul>
      <small v-else-if="detail" class="detail">{{ detail }}</small>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import HackerIcon from '@/components/HackerIcon.vue';

const props = defineProps<{
  level?: 'ok' | 'warn' | 'error';
  label: string;
  detail?: string;
  icon?: string;
}>();

const computedLevel = computed(() => props.level ?? 'ok');

const DEFAULT_ICONS: Record<'ok' | 'warn' | 'error', string> = {
  ok: 'overview',
  warn: 'rocket',
  error: 'shield',
};

const computedIcon = computed(() => props.icon ?? DEFAULT_ICONS[computedLevel.value]);

const detailLines = computed(() => {
  if (!props.detail) return [];
  const raw = props.detail.replace(/\s+/g, ' ').trim();
  if (!raw) return [];
  if (raw.includes('•')) {
    return raw.split('•').map((segment) => segment.trim()).filter(Boolean);
  }
  if (raw.includes('|')) {
    return raw.split('|').map((segment) => segment.trim()).filter(Boolean);
  }
  return [raw];
});
</script>

<style scoped>
.status-light {
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
  padding: 0.75rem 0.9rem;
  border-radius: 14px;
  border: 1px solid rgba(80, 150, 255, 0.25);
  background: rgba(14, 28, 48, 0.72);
  box-shadow: 0 12px 28px rgba(3, 12, 25, 0.3);
  max-inline-size: min(100%, 280px);
  box-sizing: border-box;
}

.status-light[data-level='ok'] {
  border-color: rgba(18, 209, 141, 0.45);
}
.status-light[data-level='warn'] {
  border-color: rgba(255, 183, 77, 0.45);
}
.status-light[data-level='error'] {
  border-color: rgba(255, 107, 107, 0.5);
}

.glyph {
  flex: 0 0 auto;
  color: rgba(255, 255, 255, 0.78);
  display: flex;
  align-items: center;
  justify-content: center;
}

.copy {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  min-width: 0;
  color: rgba(240, 245, 255, 0.85);
}

.copy strong {
  display: block;
  text-transform: uppercase;
  letter-spacing: 0.14rem;
  font-size: 0.78rem;
  color: rgba(240, 245, 255, 0.95);
}

.detail,
.detail-list {
  font-size: 0.72rem;
  color: rgba(220, 230, 255, 0.72);
  overflow-wrap: anywhere;
}

.detail-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  max-height: 4.5rem;
  overflow-y: auto;
}

.detail-list li::before {
  content: '▹';
  margin-right: 0.35rem;
  color: rgba(120, 200, 255, 0.7);
}
</style>
