<template>
  <div class="preview-chart" v-if="pointsActual.length > 1 && width > 0 && height > 0">
    <svg :viewBox="`0 0 ${width} ${height}`" role="img" aria-label="Price preview chart">
      <defs>
        <linearGradient id="actualGradient" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="rgba(79,168,255,0.6)" />
          <stop offset="100%" stop-color="rgba(79,168,255,0.05)" />
        </linearGradient>
        <linearGradient id="predictedGradient" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="rgba(255,218,97,0.6)" />
          <stop offset="100%" stop-color="rgba(255,218,97,0.05)" />
        </linearGradient>
      </defs>
      <path :d="areaActual" fill="url(#actualGradient)" stroke="none" />
      <path :d="lineActual" fill="none" stroke="rgba(79,168,255,0.85)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" />
      <path :d="linePredicted" fill="none" stroke="rgba(255,176,58,0.85)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="6 4" />
      <g v-if="selectedPoint">
        <line
          :x1="selectedPoint.x"
          :x2="selectedPoint.x"
          y1="0"
          :y2="height"
          stroke="rgba(255,255,255,0.2)"
          stroke-width="1"
        />
        <circle :cx="selectedPoint.x" :cy="selectedPoint.yActual" r="4" fill="#4FA8FF" />
        <circle :cx="selectedPoint.x" :cy="selectedPoint.yPredicted" r="4" fill="#FFB03A" />
      </g>
      <g class="axis">
        <line :x1="padding" :x2="width - padding" :y1="height - padding" :y2="height - padding" />
        <line :x1="padding" :x2="padding" :y1="padding" :y2="height - padding" />
      </g>
      <g class="ticks">
        <template v-for="tick in xTicks" :key="`x-${tick.ts}`">
          <line :x1="tick.x" :x2="tick.x" :y1="height - padding" :y2="height - padding + 6" />
          <text :x="tick.x" :y="height - padding + 18">{{ tick.label }}</text>
        </template>
        <template v-for="tick in yTicks" :key="`y-${tick.value}`">
          <line :x1="padding - 6" :x2="padding" :y1="tick.y" :y2="tick.y" />
          <text :x="padding - 8" :y="tick.y + 4">{{ tick.label }}</text>
        </template>
      </g>
    </svg>
  </div>
  <div v-else class="preview-chart empty">Insufficient data to render preview.</div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

interface SeriesPoint {
  timestamp: number;
  current_price: number;
  future_price: number;
  predicted_price: number;
}

const props = withDefaults(
  defineProps<{
    series: SeriesPoint[];
    selectedIndex?: number;
    width?: number;
    height?: number;
  }>(),
  {
    width: 820,
    height: 280,
    selectedIndex: -1,
  }
);

const padding = 36;

const domainTime = computed(() => {
  const stamps = props.series.map((point) => Number(point.timestamp || 0)).filter((val) => Number.isFinite(val));
  if (!stamps.length) return { min: 0, max: 0 };
  return { min: Math.min(...stamps), max: Math.max(...stamps) };
});

const domainPrice = computed(() => {
  const values: number[] = [];
  props.series.forEach((point) => {
    if (Number.isFinite(point.current_price)) values.push(point.current_price);
    if (Number.isFinite(point.future_price)) values.push(point.future_price);
    if (Number.isFinite(point.predicted_price)) values.push(point.predicted_price);
  });
  if (!values.length) return { min: 0, max: 1 };
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) {
    return { min: min * 0.95, max: max * 1.05 || 1 };
  }
  return { min, max };
});

const scaleX = (ts: number) => {
  const { min, max } = domainTime.value;
  if (max === min) {
    const idx = props.series.findIndex((pt) => pt.timestamp === ts);
    return padding + ((props.width - padding * 2) * idx) / Math.max(props.series.length - 1, 1);
  }
  return padding + ((ts - min) / (max - min)) * (props.width - padding * 2);
};

const scaleY = (price: number) => {
  const { min, max } = domainPrice.value;
  if (max === min) return props.height / 2;
  const ratio = (price - min) / (max - min);
  return props.height - padding - ratio * (props.height - padding * 2);
};

const pointsActual = computed(() =>
  props.series.map((point) => ({
    x: scaleX(Number(point.timestamp || 0)),
    y: scaleY(point.future_price),
  }))
);

const pointsPredicted = computed(() =>
  props.series.map((point) => ({
    x: scaleX(Number(point.timestamp || 0)),
    y: scaleY(point.predicted_price),
  }))
);

const lineActual = computed(() => {
  if (pointsActual.value.length < 2) return '';
  return pointsActual.value.map((pt, idx) => `${idx === 0 ? 'M' : 'L'} ${pt.x} ${pt.y}`).join(' ');
});

const areaActual = computed(() => {
  if (pointsActual.value.length < 2) return '';
  const baseY = props.height - padding;
  const pathStart = `M ${pointsActual.value[0].x} ${baseY}`;
  const line = pointsActual.value.map((pt) => `L ${pt.x} ${pt.y}`).join(' ');
  const pathEnd = `L ${pointsActual.value.at(-1)?.x ?? 0} ${baseY} Z`;
  return `${pathStart} ${line} ${pathEnd}`;
});

const linePredicted = computed(() => {
  if (pointsPredicted.value.length < 2) return '';
  return pointsPredicted.value.map((pt, idx) => `${idx === 0 ? 'M' : 'L'} ${pt.x} ${pt.y}`).join(' ');
});

const selectedPoint = computed(() => {
  if (props.selectedIndex === undefined || props.selectedIndex === null) return null;
  const idx = Math.min(Math.max(props.selectedIndex, 0), props.series.length - 1);
  const seriesPoint = props.series[idx];
  const actual = pointsActual.value[idx];
  const predicted = pointsPredicted.value[idx];
  if (!seriesPoint || !actual || !predicted) return null;
  return {
    x: actual.x,
    yActual: actual.y,
    yPredicted: predicted.y,
  };
});

const xTicks = computed(() => {
  const { min, max } = domainTime.value;
  if (min === max) return [];
  const span = max - min;
  const step = span / 4 || 1;
  return Array.from({ length: 5 }).map((_, idx) => {
    const ts = min + step * idx;
    const date = new Date(ts * 1000);
    const label = `${date.getUTCMonth() + 1}/${date.getUTCDate()} ${date.getUTCHours().toString().padStart(2, '0')}:${date
      .getUTCMinutes()
      .toString()
      .padStart(2, '0')}`;
    return { ts, x: scaleX(ts), label };
  });
});

const yTicks = computed(() => {
  const { min, max } = domainPrice.value;
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  const step = (max - min) / 4 || 1;
  return Array.from({ length: 5 }).map((_, idx) => {
    const value = min + step * idx;
    return {
      value,
      y: scaleY(value),
      label: value.toFixed(2),
    };
  });
});
</script>

<style scoped>
.preview-chart {
  width: 100%;
  color: rgba(240, 245, 255, 0.75);
}

.preview-chart svg {
  width: 100%;
  height: auto;
  display: block;
  overflow: visible;
}

.preview-chart .axis line {
  stroke: rgba(240, 245, 255, 0.15);
  stroke-width: 1;
}

.preview-chart .ticks line {
  stroke: rgba(240, 245, 255, 0.1);
}

.preview-chart .ticks text {
  font-size: 10px;
  fill: rgba(240, 245, 255, 0.6);
  text-anchor: middle;
}

.preview-chart .ticks text[y] {
  text-anchor: end;
}

.preview-chart.empty {
  padding: 1.5rem;
  text-align: center;
  background: rgba(6, 14, 26, 0.6);
  border: 1px dashed rgba(79, 168, 255, 0.4);
  border-radius: 12px;
}
</style>
