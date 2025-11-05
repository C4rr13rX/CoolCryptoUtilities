<template>
  <div class="console-view">
    <section class="status-panel panel">
      <header>
        <h2>Production Manager</h2>
        <span class="caption">{{ statusSummary }}</span>
      </header>
      <div class="actions">
        <button class="btn" type="button" @click="store.startProcess" :disabled="isRunning">
          Start
        </button>
        <button class="btn danger" type="button" @click="store.stopProcess" :disabled="!isRunning">
          Stop
        </button>
      </div>
      <dl class="status-grid">
        <div>
          <dt>Status</dt>
          <dd>{{ store.consoleStatus?.status ?? 'idle' }}</dd>
        </div>
        <div>
          <dt>PID</dt>
          <dd>{{ store.consoleStatus?.pid ?? '—' }}</dd>
        </div>
        <div>
          <dt>Uptime</dt>
          <dd>{{ uptimeDisplay }}</dd>
        </div>
        <div>
          <dt>Return code</dt>
          <dd>{{ store.consoleStatus?.returncode ?? '—' }}</dd>
        </div>
      </dl>
    </section>

    <section class="panel input-panel">
      <header>
        <h2>Interactive Console</h2>
        <span class="caption">Send commands directly to main.py</span>
      </header>
      <form class="command-form" @submit.prevent="sendCommand">
        <input
          v-model="command"
          type="text"
          placeholder="Type a console command (e.g. 1, 2, 7, bridge...)"
        />
        <button class="btn" type="submit" :disabled="!command.trim()">Send</button>
      </form>
      <p class="helper">
        Commands are piped to the running CLI session. Option 7 is sent automatically on start.
      </p>
    </section>

    <section class="panel log-panel">
      <header>
        <h2>Console Output</h2>
        <span class="caption">{{ logLines.length }} lines</span>
      </header>
      <pre class="console-output" ref="logContainer">{{ logLines.join('\n') }}</pre>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import { useDashboardStore } from '@/stores/dashboard';

const store = useDashboardStore();
const command = ref('');
const logContainer = ref<HTMLElement | null>(null);

const isRunning = computed(() => store.consoleStatus?.status?.includes('run'));

const uptimeDisplay = computed(() => {
  const uptime = Number(store.consoleStatus?.uptime ?? 0);
  if (!uptime) return '—';
  if (uptime < 60) return `${uptime.toFixed(1)} s`;
  if (uptime < 3600) return `${(uptime / 60).toFixed(1)} min`;
  return `${(uptime / 3600).toFixed(1)} h`;
});

const logLines = computed(() => store.consoleLogs || []);

const statusSummary = computed(() => {
  if (!store.consoleStatus) return 'Idle';
  if (store.consoleStatus.status?.includes('run')) {
    return `PID ${store.consoleStatus.pid} • up ${uptimeDisplay.value}`;
  }
  if (store.consoleStatus.returncode) {
    return `Exited (${store.consoleStatus.returncode})`;
  }
  return store.consoleStatus.status ?? 'Idle';
});

async function sendCommand() {
  const payload = command.value.trim();
  if (!payload) return;
  await store.sendConsoleInput(payload);
  command.value = '';
}

watch(
  () => logLines.value.length,
  async () => {
    await nextTick();
    const el = logContainer.value;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }
);
</script>

<style scoped>
.console-view {
  display: grid;
  grid-template-columns: 340px 1fr;
  gap: 1.5rem;
  align-items: start;
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

.caption {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
}

.status-panel {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.actions {
  display: flex;
  gap: 0.75rem;
}

.btn {
  background: linear-gradient(135deg, rgba(45, 117, 196, 0.9), rgba(12, 33, 66, 0.95));
  border: 1px solid rgba(111, 167, 255, 0.3);
  color: #f4f6fa;
  padding: 0.45rem 1.2rem;
  border-radius: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.btn.danger {
  background: linear-gradient(135deg, rgba(255, 118, 118, 0.9), rgba(133, 28, 28, 0.95));
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.8rem;
}

.status-grid dt {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1rem;
  color: rgba(255, 255, 255, 0.6);
  margin-bottom: 0.2rem;
}

.status-grid dd {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 600;
}

.input-panel {
  grid-column: span 2;
}

.command-form {
  display: flex;
  gap: 0.75rem;
}

.command-form input {
  flex: 1;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(111, 167, 255, 0.25);
  border-radius: 10px;
  padding: 0.65rem 0.9rem;
  color: #f4f6fa;
  font-family: 'Fira Code', 'Source Code Pro', monospace;
  font-size: 0.95rem;
}

.helper {
  margin-top: 0.75rem;
  color: rgba(255, 255, 255, 0.55);
  font-size: 0.8rem;
}

.log-panel {
  grid-column: span 2;
}

.console-output {
  font-family: 'Fira Code', 'Source Code Pro', monospace;
  background: rgba(8, 15, 27, 0.9);
  border-radius: 12px;
  padding: 1rem;
  max-height: 26rem;
  overflow-y: auto;
  border: 1px solid rgba(111, 167, 255, 0.18);
  white-space: pre-wrap;
}

@media (max-width: 960px) {
  .console-view {
    grid-template-columns: 1fr;
  }
  .input-panel,
  .log-panel {
    grid-column: span 1;
  }
}
</style>
