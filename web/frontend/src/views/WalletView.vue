<template>
  <div class="wallet-view">
    <section class="panel summary-panel">
      <header>
        <div>
          <h2>Wallet Overview</h2>
          <p class="caption">{{ wallet.snapshot?.wallet || 'No wallet detected' }}</p>
        </div>
        <div class="summary-actions">
          <button class="btn" type="button" @click="wallet.autoRefresh" :disabled="wallet.running || wallet.autoRefreshing">
            {{ wallet.autoRefreshing || wallet.running ? 'Refreshing…' : 'Auto Refresh' }}
          </button>
          <button class="btn ghost" type="button" @click="wallet.fetchSnapshot" :disabled="wallet.snapshotLoading">
            Reload Snapshot
          </button>
        </div>
      </header>
      <div class="summary-grid">
        <div>
          <span class="label">Total USD</span>
          <span class="value">{{ totalUsdDisplay }}</span>
        </div>
        <div>
          <span class="label">Last Updated</span>
          <span class="value">{{ snapshotTimestamp }}</span>
        </div>
        <div>
          <span class="label">Worker Status</span>
          <span class="value">{{ workerSummary }}</span>
        </div>
      </div>
    </section>

    <section class="panel actions-panel" v-if="wallet.actions.length">
      <header>
        <div>
          <h2>Wallet Actions</h2>
          <p class="caption">Live controls bound to main.py options</p>
        </div>
        <span class="caption" v-if="wallet.running">Running…</span>
      </header>
      <div class="action-tabs">
        <button
          v-for="action in wallet.actions"
          :key="action.name"
          type="button"
          class="tab"
          :class="{ active: activeAction === action.name }"
          @click="setActiveAction(action.name)"
        >
          {{ action.label }}
        </button>
      </div>
      <div v-if="currentAction" class="tab-body">
        <p class="caption">{{ currentAction.description }}</p>
        <form class="action-form" @submit.prevent="submitAction(currentAction)">
          <label v-for="field in currentAction.fields" :key="field.name">
            <span>{{ field.label }}</span>
            <select v-if="field.kind === 'select'" v-model="formState[currentAction.name][field.name]" :required="field.required">
              <option value="" disabled selected hidden>Select</option>
              <option v-for="opt in field.options" :key="opt" :value="opt">{{ opt }}</option>
            </select>
            <input
              v-else
              :type="field.kind === 'number' ? 'number' : 'text'"
              v-model="formState[currentAction.name][field.name]"
              :placeholder="field.placeholder || ''"
              :required="field.required"
            />
          </label>
          <div class="actions">
            <button class="btn" type="submit" :disabled="wallet.running">Run Action</button>
          </div>
        </form>
      </div>
    </section>

    <section class="panel balances-panel">
      <header>
        <div>
          <h2>Token Balances</h2>
          <p class="caption">Filtered by scam registry & cached for the UI</p>
        </div>
      </header>
      <div class="table-scroll" v-if="walletBalances.length">
        <table>
          <thead>
            <tr>
              <th>Chain</th>
              <th>Token</th>
              <th class="align-right">Quantity</th>
              <th class="align-right">USD</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in walletBalances" :key="row.chain + row.token">
              <td>{{ row.chain }}</td>
              <td>{{ row.symbol }}</td>
              <td class="align-right">{{ formatNumber(row.quantity) }}</td>
              <td class="align-right">{{ currency(row.usd) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="empty-text">No balances cached yet.</p>
    </section>

    <section class="panel transfers-panel">
      <header>
        <div>
          <h2>Recent Transfers</h2>
          <p class="caption">Latest 25 per chain</p>
        </div>
      </header>
      <div class="transfers-grid" v-if="transferEntries.length">
        <article v-for="[chain, items] in transferEntries" :key="chain" class="transfer-card">
          <header>
            <strong>{{ chain }}</strong>
          </header>
          <ul>
            <li v-for="item in items" :key="item.hash + item.block">
              <span class="direction" :class="item.direction">{{ item.direction === 'in' ? 'In' : 'Out' }}</span>
              <span class="amount">{{ formatValue(item.value) }}</span>
              <span class="hash">{{ truncateHash(item.hash) }}</span>
            </li>
          </ul>
        </article>
      </div>
      <p v-else class="empty-text">No transfers cached yet.</p>
    </section>

    <section class="panel nft-panel">
      <header class="nft-header">
        <div>
          <h2>NFT Holdings</h2>
          <p class="caption">Resolved via major gateways (Alchemy / IPFS / Arweave)</p>
        </div>
        <div class="nft-tabs">
          <button
            class="tab"
            type="button"
            :class="{ active: nftTab === 'shown' }"
            @click="nftTab = 'shown'"
          >
            Shown ({{ shownNfts.length }})
          </button>
          <button
            class="tab"
            type="button"
            :class="{ active: nftTab === 'hidden' }"
            @click="nftTab = 'hidden'"
          >
            Hidden ({{ hiddenNfts.length }})
          </button>
        </div>
      </header>
      <div class="nft-actions">
        <button
          v-if="nftTab === 'shown'"
          class="btn ghost"
          type="button"
          :disabled="!shownSelectionCount"
          @click="hideSelectedNfts"
        >
          Hide Selected
        </button>
        <button
          v-else
          class="btn ghost"
          type="button"
          :disabled="!hiddenSelectionCount"
          @click="showSelectedNfts"
        >
          Show Selected
        </button>
      </div>
      <div class="nft-grid" v-if="activeNfts.length">
        <article v-for="nft in activeNfts" :key="nftKey(nft)" class="nft-card">
          <label class="nft-select">
            <input
              type="checkbox"
              :checked="isNftSelected(nft)"
              @change="toggleNftSelection(nft)"
            />
            <span class="checkmark"></span>
          </label>
          <div class="thumb" :class="{ placeholder: !nft.image }">
            <img v-if="nft.image" :src="nft.image" :alt="nft.title || 'NFT'" loading="lazy" />
            <span v-else>No image</span>
          </div>
          <div class="nft-meta">
            <strong>{{ nft.title || 'Untitled' }}</strong>
            <span class="caption">{{ nft.chain }}</span>
            <span class="token-id">#{{ nft.token_id }}</span>
          </div>
        </article>
      </div>
      <p v-else class="empty-text">
        {{ nftTab === 'shown' ? 'No NFTs shown yet.' : 'No hidden NFTs.' }}
      </p>
    </section>

    <section class="panel console-panel">
      <header>
        <div>
          <h2>Automation Consoles</h2>
          <p class="caption">Mirrors guardian + production manager logs.</p>
        </div>
        <button class="btn ghost" type="button" @click="refreshConsoleLogs" :disabled="consoleBusy">
          {{ consoleBusy ? 'Refreshing…' : 'Refresh Logs' }}
        </button>
      </header>
      <AutomationConsoleStack :manager-lines="consoleLines" :guardian-lines="guardianConsole" />
    </section>

    <section class="panel mnemonic-panel">
      <header>
        <div>
          <h2>Wallet Seed</h2>
          <span class="caption">MNEMONIC stored in the secure vault</span>
        </div>
        <span class="caption">{{ wallet.mnemonicPreview ? 'Loaded' : 'Not set' }}</span>
      </header>
      <form class="mnemonic-form" @submit.prevent="saveMnemonic">
        <textarea v-model="mnemonicInput" rows="3" placeholder="problem tube idea ..."></textarea>
        <div class="actions">
          <button class="btn" type="submit">Save</button>
          <button class="btn ghost" type="button" @click="clearMnemonic">Clear</button>
        </div>
      </form>
    </section>
  </div>
  <TradingStartupWizard
    v-if="wizardSteps.length"
    v-model:open="wizardOpen"
    :steps="wizardSteps"
    title="Trading Launch Checklist"
    subtitle="Finish the missing wallet steps to unlock ghost + live trading."
    eyebrow="Wallet Wizard"
  >
    <template #step-mnemonic>
      <div class="wizard-field">
        <label for="wizard-mnemonic">Recovery phrase</label>
        <textarea
          id="wizard-mnemonic"
          v-model="wizardMnemonicInput"
          rows="3"
          placeholder="problem tube idea ..."
        ></textarea>
        <div class="wizard-actions">
          <button class="btn" type="button" @click="saveWizardMnemonic" :disabled="!wizardMnemonicInput.trim()">
            Save Phrase
          </button>
          <button class="btn ghost" type="button" @click="clearWizardMnemonic">
            Clear
          </button>
        </div>
      </div>
    </template>
  </TradingStartupWizard>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue';
import { fetchWalletNftPreferences, updateWalletNftPreferences } from '@/api';
import { useDashboardStore } from '@/stores/dashboard';
import { useWalletStore } from '@/stores/wallet';
import AutomationConsoleStack from '@/components/AutomationConsoleStack.vue';
import TradingStartupWizard from '@/components/TradingStartupWizard.vue';

const dashboard = useDashboardStore();
const wallet = useWalletStore();
const mnemonicInput = ref('');
const formState = reactive<Record<string, Record<string, string>>>({});
const activeAction = ref('');
const statusTimer = ref<number>();
const consoleTimer = ref<number>();
const consoleBusy = ref(false);
const wizardOpen = ref(true);
const wizardMnemonicInput = ref('');
const nftTab = ref<'shown' | 'hidden'>('shown');
const nftHidden = ref<Set<string>>(new Set());
const nftSelectionShown = ref<Set<string>>(new Set());
const nftSelectionHidden = ref<Set<string>>(new Set());

const walletBalances = computed(() => wallet.balances);
const transferEntries = computed(() => Object.entries(wallet.transfers || {}));
const consoleLines = computed(() => dashboard.consoleLogs || []);
const guardianConsole = computed(() => dashboard.guardianLogs || []);
const totalUsdDisplay = computed(() => currency(wallet.snapshot?.totals?.usd || 0));
const snapshotTimestamp = computed(() => wallet.snapshot?.updated_at || 'Never');
const shownNfts = computed(() => wallet.nfts.filter((nft: any) => !nftHidden.value.has(nftKey(nft))));
const hiddenNfts = computed(() => wallet.nfts.filter((nft: any) => nftHidden.value.has(nftKey(nft))));
const activeNfts = computed(() => (nftTab.value === 'shown' ? shownNfts.value : hiddenNfts.value));
const shownSelectionCount = computed(() => nftSelectionShown.value.size);
const hiddenSelectionCount = computed(() => nftSelectionHidden.value.size);

const workerSummary = computed(() => {
  if (wallet.running || wallet.autoRefreshing || wallet.status?.running) {
    return 'Updating…';
  }
  const status = wallet.status || {};
  const message = String(status.message || 'idle');
  const returncode = status.returncode;
  if (message.toLowerCase().startsWith('failed') || (typeof returncode === 'number' && returncode !== 0)) {
    const finishedAt = Number(status.finished_at || 0);
    const snapshotAt = parseTimestamp(wallet.snapshot?.updated_at);
    if (snapshotAt && finishedAt && snapshotAt > (finishedAt * 1000 + 5000)) {
      return 'Updated (last run failed)';
    }
    return message;
  }
  return message === 'idle' ? 'Idle' : message;
});

const currentAction = computed(() => wallet.actions.find((action: any) => action.name === activeAction.value));

type WizardStep = {
  id: string;
  title: string;
  description: string;
  detail?: string;
  ctaLabel?: string;
  ctaAction?: () => void;
  tone?: 'info' | 'warning' | 'critical' | 'success';
};

const wizardSteps = computed<WizardStep[]>(() => {
  const steps: WizardStep[] = [];
  const hasMnemonic = Boolean(wallet.mnemonicPreview);
  const productionKnown = wallet.status !== null;
  const productionRunning = Boolean(wallet.status?.production?.running);

  if (!hasMnemonic) {
    steps.push({
      id: 'mnemonic',
      title: 'Add wallet recovery phrase',
      description: 'Live trading needs your wallet seed stored in the secure vault.',
      detail: 'This populates the MNEMONIC secret for automated signing.',
      tone: 'critical',
    });
  }

  if (productionKnown && !productionRunning) {
    steps.push({
      id: 'production',
      title: 'Start live trading engine',
      description: 'The production manager is offline, so trades cannot execute yet.',
      detail: 'Kick off the production supervisor once the wallet + readiness gates are clear.',
      ctaLabel: 'Start Production',
      ctaAction: () => startProduction(),
      tone: 'warning',
    });
  }

  return steps;
});

function ensureForm(action: any) {
  if (!formState[action.name]) {
    formState[action.name] = {};
    (action.fields || []).forEach((field: any) => {
      formState[action.name][field.name] = '';
    });
  }
  return formState[action.name];
}

function setActiveAction(name: string) {
  activeAction.value = name;
  const action = wallet.actions.find((entry: any) => entry.name === name);
  if (action) {
    ensureForm(action);
  }
}

async function submitAction(action: any) {
  const payload = action.fields?.length ? { ...ensureForm(action) } : undefined;
  try {
    await wallet.run(action.name, payload);
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn('Wallet action failed', error);
  }
}

async function saveMnemonic() {
  const value = mnemonicInput.value.trim();
  await wallet.saveMnemonic(value || null);
  mnemonicInput.value = '';
}

async function clearMnemonic() {
  mnemonicInput.value = '';
  await wallet.saveMnemonic(null);
}

async function saveWizardMnemonic() {
  const value = wizardMnemonicInput.value.trim();
  if (!value) return;
  await wallet.saveMnemonic(value);
  wizardMnemonicInput.value = '';
}

async function clearWizardMnemonic() {
  wizardMnemonicInput.value = '';
  await wallet.saveMnemonic(null);
}

async function startProduction() {
  if (wallet.running) return;
  try {
    await wallet.run('start_production');
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn('Failed to start production manager', error);
  }
}

async function refreshConsoleLogs() {
  if (consoleBusy.value) return;
  consoleBusy.value = true;
  try {
    await dashboard.refreshConsole();
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn('Failed to refresh console logs', error);
  } finally {
    consoleBusy.value = false;
  }
}

onMounted(async () => {
  await Promise.all([
    wallet.refreshStatus(),
    wallet.fetchSnapshot(),
    wallet.loadMnemonicPreview(),
    dashboard.refreshConsole(),
  ]);
  await loadNftPreferences();
  wallet.autoRefresh();
  statusTimer.value = window.setInterval(() => wallet.refreshStatus(), 6000);
  consoleTimer.value = window.setInterval(() => dashboard.refreshConsole().catch(() => undefined), 10000);
});

onBeforeUnmount(() => {
  if (statusTimer.value) {
    window.clearInterval(statusTimer.value);
  }
  if (consoleTimer.value) {
    window.clearInterval(consoleTimer.value);
  }
});

watch(
  () => wallet.actions,
  (actions) => {
    if (actions && actions.length && !activeAction.value) {
      activeAction.value = actions[0].name;
    }
    actions?.forEach((action: any) => ensureForm(action));
  },
  { immediate: true }
);

watch(
  () => wallet.status?.running,
  (running, prev) => {
    if (prev && !running) {
      wallet.fetchSnapshot();
    }
  }
);

watch(
  () => wallet.nfts,
  () => {
    pruneSelections();
  }
);

function currency(value: number | string | undefined) {
  const num = Number(value || 0);
  return Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(num);
}

function formatNumber(value: number | string) {
  const num = Number(value || 0);
  if (Math.abs(num) >= 1) {
    return num.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }
  return num.toPrecision(4);
}

function formatValue(value: any) {
  if (!value) return '—';
  if (typeof value === 'string' && value.startsWith('0x')) {
    return parseInt(value, 16).toString();
  }
  return value;
}

function truncateHash(hash: string | undefined) {
  if (!hash) return 'unknown';
  return `${hash.slice(0, 6)}…${hash.slice(-4)}`;
}

function parseTimestamp(value: unknown): number {
  if (!value) return 0;
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
}

function nftKey(nft: any): string {
  const chain = String(nft?.chain || '').toLowerCase();
  const contract = String(nft?.contract || '').toLowerCase();
  const tokenId = String(nft?.token_id || '').trim();
  return `${chain}:${contract}:${tokenId}`;
}

function selectionForTab(tab: 'shown' | 'hidden') {
  return tab === 'shown' ? nftSelectionShown : nftSelectionHidden;
}

function isNftSelected(nft: any): boolean {
  const set = selectionForTab(nftTab.value).value;
  return set.has(nftKey(nft));
}

function toggleNftSelection(nft: any) {
  const key = nftKey(nft);
  const selectionRef = selectionForTab(nftTab.value);
  const next = new Set(selectionRef.value);
  if (next.has(key)) {
    next.delete(key);
  } else {
    next.add(key);
  }
  selectionRef.value = next;
}

function pruneSelections() {
  const valid = new Set(wallet.nfts.map((nft: any) => nftKey(nft)));
  const prune = (selectionRef: typeof nftSelectionShown) => {
    const next = new Set<string>();
    selectionRef.value.forEach((key) => {
      if (valid.has(key)) next.add(key);
    });
    selectionRef.value = next;
  };
  prune(nftSelectionShown);
  prune(nftSelectionHidden);
}

async function loadNftPreferences() {
  try {
    const data = await fetchWalletNftPreferences();
    const hiddenKeys = new Set((data.items || []).map((item) => nftKey(item)));
    nftHidden.value = hiddenKeys;
  } catch (error) {
    // ignore preference failures for now
  }
}

async function updateNftVisibility(action: 'hide' | 'show', items: any[]) {
  if (!items.length) return;
  const payload = items.map((nft) => ({
    chain: String(nft.chain || '').toLowerCase(),
    contract: String(nft.contract || '').toLowerCase(),
    token_id: String(nft.token_id || ''),
  }));
  try {
    const data = await updateWalletNftPreferences({ action, items: payload });
    nftHidden.value = new Set((data.items || []).map((item) => nftKey(item)));
  } catch (error) {
    // ignore preference failures for now
  }
}

async function hideSelectedNfts() {
  const selected = shownNfts.value.filter((nft: any) => nftSelectionShown.value.has(nftKey(nft)));
  await updateNftVisibility('hide', selected);
  nftSelectionShown.value = new Set();
}

async function showSelectedNfts() {
  const selected = hiddenNfts.value.filter((nft: any) => nftSelectionHidden.value.has(nftKey(nft)));
  await updateNftVisibility('show', selected);
  nftSelectionHidden.value = new Set();
}
</script>

<style scoped>
.wallet-view {
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
  align-items: flex-start;
  gap: 1rem;
  margin-bottom: 1rem;
}

.caption {
  font-size: 0.82rem;
  color: rgba(255, 255, 255, 0.65);
}

.summary-actions {
  display: flex;
  gap: 0.6rem;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
}

.summary-grid .label {
  display: block;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.1rem;
  color: rgba(255, 255, 255, 0.6);
}

.summary-grid .value {
  font-size: 1.2rem;
  font-weight: 600;
}

.actions-panel .action-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-bottom: 1rem;
}

.action-tabs .tab {
  border-radius: 999px;
  border: 1px solid rgba(111, 167, 255, 0.3);
  background: transparent;
  color: #f4f6fa;
  padding: 0.45rem 1rem;
  cursor: pointer;
}

.action-tabs .tab.active {
  background: rgba(45, 117, 196, 0.35);
  border-color: rgba(45, 117, 196, 0.65);
}

.action-form {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.9rem;
}

.action-form label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.action-form input,
.action-form select,
.mnemonic-form textarea {
  border-radius: 10px;
  border: 1px solid rgba(111, 167, 255, 0.25);
  background: rgba(255, 255, 255, 0.04);
  color: #f4f6fa;
  padding: 0.6rem 0.7rem;
}

.balances-panel .table-scroll {
  border-radius: 16px;
  border-color: rgba(111, 167, 255, 0.2);
}

.align-right {
  text-align: right;
}

.empty-text {
  color: rgba(255, 255, 255, 0.55);
  font-style: italic;
}

.transfers-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
}

.transfer-card {
  border: 1px solid rgba(111, 167, 255, 0.18);
  border-radius: 12px;
  padding: 0.9rem;
  background: rgba(7, 14, 25, 0.85);
}

.transfer-card ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.transfer-card li {
  display: flex;
  justify-content: space-between;
  gap: 0.4rem;
  font-size: 0.85rem;
}

.direction {
  font-weight: 600;
  text-transform: uppercase;
}

.direction.in {
  color: #34d399;
}

.direction.out {
  color: #f87171;
}

.hash {
  color: rgba(255, 255, 255, 0.6);
}

.nft-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.nft-header {
  align-items: center;
}

.nft-tabs {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.nft-tabs .tab {
  border-radius: 999px;
  border: 1px solid rgba(111, 167, 255, 0.3);
  background: transparent;
  color: #f4f6fa;
  padding: 0.35rem 0.9rem;
  cursor: pointer;
}

.nft-tabs .tab.active {
  background: rgba(45, 117, 196, 0.35);
  border-color: rgba(45, 117, 196, 0.65);
}

.nft-actions {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 0.8rem;
}

.nft-card {
  border: 1px solid rgba(111, 167, 255, 0.18);
  border-radius: 16px;
  overflow: hidden;
  background: rgba(7, 14, 25, 0.85);
  position: relative;
}

.nft-select {
  position: absolute;
  top: 0.6rem;
  left: 0.6rem;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  z-index: 2;
}

.nft-select input {
  opacity: 0;
  position: absolute;
  width: 0;
  height: 0;
}

.nft-select .checkmark {
  width: 16px;
  height: 16px;
  border-radius: 4px;
  border: 1px solid rgba(111, 167, 255, 0.6);
  background: rgba(7, 14, 25, 0.8);
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.nft-select input:checked + .checkmark {
  background: rgba(69, 149, 245, 0.6);
  border-color: rgba(99, 167, 255, 0.9);
}

.nft-card .thumb {
  height: 180px;
  background: rgba(255, 255, 255, 0.04);
  display: flex;
  align-items: center;
  justify-content: center;
}

.nft-card img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.nft-card .thumb.placeholder {
  color: rgba(255, 255, 255, 0.6);
}

.nft-meta {
  padding: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.console-panel header {
  align-items: center;
}

.mnemonic-form {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.actions {
  display: flex;
  gap: 0.8rem;
}

.wizard-field {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.wizard-field label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.14rem;
  color: rgba(255, 255, 255, 0.7);
}

.wizard-field textarea {
  border-radius: 12px;
  border: 1px solid rgba(127, 176, 255, 0.3);
  background: rgba(255, 255, 255, 0.04);
  color: #f4f6fa;
  padding: 0.6rem 0.7rem;
}

.wizard-actions {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
}
</style>
