<template>
  <div class="ticker" v-if="items.length">
    <div class="ticker__track" :style="trackStyle">
      <div class="ticker__item" v-for="item in doubledItems" :key="item.key">
        <span :class="['badge', severityClass(item.severity)]">{{ item.symbol }}</span>
        <span class="price">{{ formatNumber(item.price) }}</span>
        <span class="meta">{{ item.meta }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

interface TickerItem {
  symbol: string;
  price: number;
  meta: string;
  severity: 'ok' | 'warn' | 'error';
}

const props = defineProps<{
  items: TickerItem[];
}>();

const doubledItems = computed(() =>
  [...props.items, ...props.items].map((item, idx) => ({
    ...item,
    key: `${item.symbol}-${idx}`,
  }))
);

const trackStyle = computed(() => ({
  animationDuration: `${Math.max(props.items.length * 6, 18)}s`,
}));

function severityClass(level: string) {
  if (level === 'error') return 'is-error';
  if (level === 'warn') return 'is-warn';
  return 'is-ok';
}

function formatNumber(value: number) {
  if (!Number.isFinite(value)) return '—';
  if (value >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (value >= 1) return value.toFixed(4);
  return value.toPrecision(4);
}
</script>

<style scoped>
.ticker {
  position: relative;
  overflow: hidden;
  border-radius: 14px;
  border: 1px solid rgba(111, 167, 255, 0.2);
  background: rgba(12, 23, 40, 0.75);
  box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.25);
}

.ticker__track {
  display: flex;
  width: max-content;
  animation: scroll linear infinite;
}

.ticker__item {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.65rem 1.2rem;
  white-space: nowrap;
  border-right: 1px solid rgba(111, 167, 255, 0.15);
}

.ticker__item:last-child {
  border-right: none;
}

.badge {
  text-transform: uppercase;
  font-size: 0.68rem;
  letter-spacing: 0.14rem;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.15);
}

.badge.is-ok {
  background: rgba(18, 209, 141, 0.18);
  color: #12d18d;
  border-color: rgba(18, 209, 141, 0.35);
}

.badge.is-warn {
  background: rgba(255, 183, 77, 0.18);
  color: #ffb74d;
  border-color: rgba(255, 183, 77, 0.35);
}

.badge.is-error {
  background: rgba(255, 107, 107, 0.18);
  color: #ff6b6b;
  border-color: rgba(255, 107, 107, 0.35);
}

.price {
  font-weight: 600;
  color: #f7fbff;
}

.meta {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.55);
}

@keyframes scroll {
  0% {
    transform: translateX(0);
  }
  100% {
    transform: translateX(-50%);
  }
}
</style>
