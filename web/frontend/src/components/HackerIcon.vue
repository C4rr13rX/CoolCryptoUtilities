<template>
  <svg
    class="hacker-icon"
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.6"
    stroke-linecap="round"
    stroke-linejoin="round"
    role="presentation"
  >
    <template v-for="(d, index) in paths" :key="d + index">
      <path :d="d" />
    </template>
  </svg>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  name: string;
  size?: number;
}>();

const size = computed(() => props.size ?? 22);

const ICON_PATHS: Record<string, string[]> = {
  overview: [
    'M2.458 12C3.732 7.943 7.523 5 12 5c4.477 0 8.268 2.943 9.542 7-1.274 4.057-5.065 7-9.542 7-4.477 0-8.268-2.943-9.542-7z',
    'M15 12a3 3 0 11-6 0 3 3 0 016 0z',
  ],
  organism: [
    'M4.5 4.5h6v6h-6v-6z',
    'M13.5 4.5h6v6h-6v-6z',
    'M4.5 13.5h6v6h-6v-6z',
    'M13.5 13.5h6v6h-6v-6z',
  ],
  pipeline: [
    'M3 7.5a4.5 4.5 0 014.5-4.5H9',
    'M9 3v4.125a1.125 1.125 0 001.125 1.125H13.5a3 3 0 013 3v0a3 3 0 01-3 3h-3.375A1.125 1.125 0 009 15.375V21',
    'M21 16.5a4.5 4.5 0 01-4.5 4.5H15',
  ],
  streams: [
    'M6.75 5.25a9 9 0 0110.5 0',
    'M7.5 11.25a5.25 5.25 0 019 0',
    'M9.75 16.5a1.5 1.5 0 012.25-1.3 1.5 1.5 0 012.25 1.3',
  ],
  telemetry: [
    'M3 12l3.75-7.5L12 15l3.75-5.25L21 12',
    'M12 15v6',
  ],
  console: [
    'M4.5 6.75h15a1.5 1.5 0 011.5 1.5v9a1.5 1.5 0 01-1.5 1.5h-15a1.5 1.5 0 01-1.5-1.5v-9a1.5 1.5 0 011.5-1.5z',
    'M6.75 9.75l2.25 2.25-2.25 2.25',
    'M11.25 14.25h3',
  ],
  advisories: [
    'M12 3l8.25 14.25H3.75L12 3z',
    'M12 9v3.75',
    'M12 16.5h.008',
  ],
  guardian: [
    'M12 3l7 3v5c0 4.8-3.2 9-7 9s-7-4.2-7-9V6l7-3z',
    'M9 11.5l2.5 2.5 4-4',
  ],
  datalab: [
    'M4.5 6c0-1.657 3.357-3 7.5-3s7.5 1.343 7.5 3-3.357 3-7.5 3-7.5-1.343-7.5-3z',
    'M4.5 6v6c0 1.657 3.357 3 7.5 3s7.5-1.343 7.5-3V6',
    'M4.5 12v6c0 1.657 3.357 3 7.5 3s7.5-1.343 7.5-3v-6',
  ],
  lab: [
    'M9.75 3.75h4.5',
    'M9.75 3.75v4.086a2.25 2.25 0 01-.66 1.591l-2.03 2.03a5.25 5.25 0 003.712 8.955h2.456a5.25 5.25 0 003.712-8.955l-2.03-2.03a2.25 2.25 0 01-.66-1.591V3.75',
  ],
  brain: [
    'M8.25 6.75a3 3 0 00-3 3v0a3 3 0 003 3h.75v-6h-.75z',
    'M8.25 6.75a4.5 4.5 0 014.5-4.5h.75v9h-.75a4.5 4.5 0 01-4.5-4.5z',
    'M15.75 15.75h-.75a4.5 4.5 0 01-4.5-4.5v-9h.75a4.5 4.5 0 014.5 4.5v9z',
  ],
  shield: [
    'M12 3l7.5 3v6c0 4.8-3.4 9-7.5 9s-7.5-4.2-7.5-9V6l7.5-3z',
  ],
  radar: [
    'M12 19.5a7.5 7.5 0 100-15 7.5 7.5 0 000 15z',
    'M12 12.75l3 3',
    'M12 6.75v4.5',
  ],
  rocket: [
    'M4.5 14.25s1.5-3 7.5-3 7.5 3 7.5 3-3 4.5-7.5 4.5-7.5-4.5-7.5-4.5z',
    'M12 6V3',
    'M9.75 10.5c.75-1.5 3.75-1.5 4.5 0',
  ],
  lightning: [
    'M13.5 4.5H18L10.5 21v-7.5H6L13.5 4.5z',
  ],
  terminal: [
    'M3.75 4.5h16.5a1.5 1.5 0 011.5 1.5v12a1.5 1.5 0 01-1.5 1.5H3.75a1.5 1.5 0 01-1.5-1.5v-12a1.5 1.5 0 011.5-1.5z',
    'M7.5 8.25l3 3-3 3',
    'M12.75 15.75h4.5',
  ],
  activity: [
    'M3 12l4.5-7.5L12 15l4.5-7.5L21 12',
  ],
};

const paths = computed(() => ICON_PATHS[props.name] || ICON_PATHS['overview']);
</script>

<style scoped>
.hacker-icon {
  color: inherit;
  display: block;
}
</style>
