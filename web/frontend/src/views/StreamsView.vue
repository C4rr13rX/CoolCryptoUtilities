<template>
  <div class="streams-view">
    <header class="view-header">
      <h1>Market Streams</h1>
      <p>Live consensus from the active data stream endpoints.</p>
    </header>
    <StreamsPanel :streams="store.streams" />
    <section class="stream-table panel">
      <header>
        <h2>Stream Snapshot</h2>
        <span class="caption">{{ streamRows.length }} pairs</span>
      </header>
      <table class="table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Chain</th>
            <th>Price</th>
            <th>Volume</th>
            <th>Updated</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in streamRows" :key="row.symbol">
            <td>{{ row.symbol }}</td>
            <td>{{ row.chain }}</td>
            <td>{{ row.priceDisplay }}</td>
            <td>{{ row.volumeDisplay }}</td>
            <td>{{ row.updated }}</td>
            <td>{{ row.source }}</td>
          </tr>
          <tr v-if="streamRows.length === 0">
            <td colspan="6">No live data streams are currently available.</td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import StreamsPanel from '@/components/StreamsPanel.vue';
import { useDashboardStore } from '@/stores/dashboard';

const store = useDashboardStore();

const numberFormatter = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 3,
  maximumFractionDigits: 6,
});

const volumeFormatter = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 4,
});

const streamRows = computed(() => {
  const entries = Object.entries(store.streams || {});
  return entries.map(([symbol, payload]: [string, any]) => {
    const price = Number(payload?.price ?? 0);
    const volume = Number(payload?.volume ?? 0);
    const ts = Number(payload?.ts ?? 0);
    const updated = Number.isFinite(ts)
      ? new Date(ts * 1000).toLocaleTimeString([], { hour12: false })
      : '—';
    return {
      symbol,
      chain: payload?.chain ?? '—',
      priceDisplay: Number.isFinite(price) ? numberFormatter.format(price) : payload?.price ?? '—',
      volumeDisplay: Number.isFinite(volume) ? volumeFormatter.format(volume) : payload?.volume ?? '—',
      updated,
      source: payload?.source ?? '—',
    };
  });
});
</script>

<style scoped>
.streams-view {
  display: flex;
  flex-direction: column;
  gap: 1.6rem;
}

.view-header h1 {
  margin: 0;
  font-size: 1.4rem;
  text-transform: uppercase;
  letter-spacing: 0.18rem;
  color: #6fa7ff;
}

.view-header p {
  margin: 0.25rem 0 0;
  color: rgba(255, 255, 255, 0.65);
  font-size: 0.95rem;
}

.panel {
  background: rgba(11, 22, 37, 0.85);
  border: 1px solid rgba(111, 167, 255, 0.18);
  border-radius: 18px;
  padding: 1.2rem 1.4rem;
  box-shadow: 0 20px 46px rgba(0, 0, 0, 0.32);
}

.panel header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1rem;
  margin-bottom: 1rem;
}

.panel h2 {
  margin: 0;
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.14rem;
  color: #6fa7ff;
}

.panel .caption {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
}
</style>
