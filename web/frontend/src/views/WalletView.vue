<template>
  <div class="wallet-view">
    <!-- Wallet Tabs -->
    <section class="panel wallet-tabs-panel" v-if="wallet.walletCount > 1 || wallet.walletList.length > 1">
      <div class="wallet-tab-bar">
        <button
          class="wallet-tab"
          :class="{ active: wallet.activeWalletIndex === -1 }"
          type="button"
          @click="wallet.setActiveWallet(-1)"
        >
          All Wallets
        </button>
        <button
          v-for="w in wallet.walletList"
          :key="w.index"
          class="wallet-tab"
          :class="{ active: wallet.activeWalletIndex === w.index }"
          type="button"
          @click="wallet.setActiveWallet(w.index)"
        >
          Wallet {{ w.index }}
        </button>
        <button
          class="wallet-tab create-tab"
          type="button"
          @click="handleCreateWallet"
          :disabled="creatingWallet"
        >
          + New
        </button>
      </div>
    </section>

    <section class="panel summary-panel">
      <header>
        <div>
          <h2>{{ wallet.activeWalletIndex === -1 ? t('wallet.title') : `Wallet ${wallet.activeWalletIndex}` }}</h2>
          <p class="caption">{{ activeWalletCaption }}</p>
        </div>
        <div class="summary-actions">
          <button class="btn" type="button" @click="wallet.autoRefresh" :disabled="wallet.running || wallet.autoRefreshing">
            {{ wallet.autoRefreshing || wallet.running ? t('common.refreshing') : t('wallet.auto_refresh') }}
          </button>
          <button class="btn ghost" type="button" @click="refreshAll" :disabled="wallet.snapshotLoading || wallet.multiWalletLoading">
            {{ t('wallet.reload_snapshot') }}
          </button>
        </div>
      </header>
      <div class="summary-grid">
        <div>
          <span class="label">{{ wallet.activeWalletIndex === -1 && wallet.walletCount > 1 ? 'Total USD (All Wallets)' : t('wallet.total_usd') }}</span>
          <span class="value highlight">{{ currency(wallet.activeWalletUsd) }}</span>
        </div>
        <div v-if="wallet.activeWalletIndex === -1 && wallet.walletCount > 1">
          <span class="label">Wallets</span>
          <span class="value">{{ wallet.walletCount }}</span>
        </div>
        <div>
          <span class="label">{{ t('wallet.last_updated') }}</span>
          <span class="value">{{ snapshotTimestamp }}</span>
        </div>
        <div>
          <span class="label">{{ t('wallet.worker_status') }}</span>
          <span class="value">{{ workerSummary }}</span>
        </div>
      </div>
      <!-- Reveal mnemonic button (per-wallet view) -->
      <div class="reveal-row" v-if="wallet.activeWalletIndex >= 0">
        <button class="btn ghost" type="button" @click="handleRevealMnemonic(wallet.activeWalletIndex)">
          Reveal Recovery Phrase
        </button>
      </div>
      <!-- Per-wallet breakdown when viewing All Wallets -->
      <div class="wallet-breakdown" v-if="wallet.activeWalletIndex === -1 && wallet.walletList.length > 1">
        <div class="breakdown-row" v-for="w in wallet.walletList" :key="w.index">
          <span class="breakdown-label">
            <button class="link-btn" type="button" @click="wallet.setActiveWallet(w.index)">
              Wallet {{ w.index }}
            </button>
            <span class="breakdown-addr">{{ truncateAddress(w.wallet) }}</span>
          </span>
          <span class="breakdown-actions">
            <button class="btn-tiny ghost" type="button" @click="handleRevealMnemonic(w.index)">Reveal</button>
            <span class="breakdown-value">{{ currency(w.usd) }}</span>
          </span>
        </div>
      </div>
    </section>

    <!-- Multi-wallet config info -->
    <section class="panel config-panel" v-if="wallet.multiWalletConfig && wallet.activeWalletIndex === -1">
      <div class="config-grid">
        <div>
          <span class="label">Auto-Create</span>
          <span class="value">{{ wallet.multiWalletConfig.enabled ? 'Enabled' : 'Disabled' }}</span>
        </div>
        <div>
          <span class="label">New Wallet Threshold</span>
          <span class="value">{{ currency(wallet.multiWalletConfig.threshold) }}</span>
        </div>
        <div>
          <span class="label">Max Per Wallet</span>
          <span class="value">{{ currency(wallet.multiWalletConfig.max_balance) }}</span>
        </div>
      </div>
    </section>

    <section class="panel actions-panel" v-if="wallet.actions.length">
      <header>
        <div>
          <h2>{{ t('wallet.actions_title') }}</h2>
          <p class="caption">{{ t('wallet.actions_caption') }}</p>
        </div>
        <span class="caption" v-if="wallet.running">{{ t('wallet.running') }}</span>
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
              <option value="" disabled selected hidden>{{ t('common.select') }}</option>
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
            <button class="btn" type="submit" :disabled="wallet.running">{{ t('wallet.run_action') }}</button>
          </div>
        </form>
      </div>
    </section>

    <section class="panel balances-panel">
      <header>
        <div>
          <h2>{{ t('wallet.token_balances') }}</h2>
          <p class="caption">{{ wallet.activeWalletIndex === -1 && wallet.walletCount > 1 ? 'Aggregated across all wallets' : t('wallet.balances_caption') }}</p>
        </div>
      </header>
      <div class="table-scroll" v-if="walletBalances.length">
        <table>
          <thead>
            <tr>
              <th v-if="wallet.activeWalletIndex === -1 && wallet.walletCount > 1">Wallet</th>
              <th>{{ t('common.chain') }}</th>
              <th>{{ t('wallet.token') }}</th>
              <th class="align-right">{{ t('wallet.quantity') }}</th>
              <th class="align-right">{{ t('common.usd') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in walletBalances" :key="(row.wallet_index ?? '') + row.chain + row.token">
              <td v-if="wallet.activeWalletIndex === -1 && wallet.walletCount > 1">{{ row.wallet_index ?? 0 }}</td>
              <td>{{ row.chain }}</td>
              <td>{{ row.symbol }}</td>
              <td class="align-right">{{ formatNumber(row.quantity) }}</td>
              <td class="align-right">{{ currency(row.usd) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="empty-text">{{ t('wallet.no_balances') }}</p>
    </section>

    <section class="panel transfers-panel">
      <header>
        <div>
          <h2>{{ t('wallet.recent_transfers') }}</h2>
          <p class="caption">{{ t('wallet.transfers_caption') }}</p>
        </div>
        <button class="btn ghost" type="button" @click="openAllTransactions">
          Show All Transactions
        </button>
      </header>
      <div class="transfers-grid" v-if="transferEntries.length">
        <article v-for="[chain, items] in transferEntries" :key="chain" class="transfer-card">
          <header>
            <strong>{{ chainDisplayName(chain) }}</strong>
            <span class="transfer-count">{{ items.length }}</span>
          </header>
          <ul>
            <li v-for="item in items.slice(0, 5)" :key="item.hash + item.block">
              <span class="direction" :class="item.direction">{{ item.direction === 'in' ? t('wallet.in') : t('wallet.out') }}</span>
              <span class="amount">{{ formatValue(item.value) }}</span>
              <span class="hash">{{ truncateHash(item.hash) }}</span>
            </li>
            <li v-if="items.length > 5" class="more-link">
              <button class="link-btn" type="button" @click="openAllTransactions">
                +{{ items.length - 5 }} more
              </button>
            </li>
          </ul>
        </article>
      </div>
      <p v-else class="empty-text">{{ t('wallet.no_transfers') }}</p>
    </section>

    <!-- All Transactions Modal -->
    <Teleport to="body">
      <div v-if="txModalOpen" class="modal-overlay" @click.self="closeTxModal">
        <div class="modal-card tx-modal">
          <header class="modal-header">
            <h3>All Transactions</h3>
            <button class="modal-close" type="button" @click="closeTxModal">&times;</button>
          </header>
          <div class="tx-toolbar">
            <input
              v-model="txSearch"
              class="tx-search-input"
              type="text"
              placeholder="Search transactions (hash, token, date, chain...)"
              @input="debouncedTxSearch"
            />
            <button
              class="btn-tiny"
              type="button"
              @click="toggleTxSort"
            >
              {{ txSort === 'desc' ? 'Latest First' : 'Earliest First' }}
            </button>
          </div>
          <div class="tx-scroll" ref="txScrollRef" @scroll="handleTxScroll">
            <table v-if="txItems.length" class="tx-table">
              <thead>
                <tr>
                  <th>Chain</th>
                  <th>Dir</th>
                  <th>Value</th>
                  <th>Token</th>
                  <th>Block</th>
                  <th>Hash</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(item, idx) in txItems" :key="idx">
                  <td>{{ chainDisplayName(item.chain) }}</td>
                  <td>
                    <span class="direction" :class="item.direction">{{ item.direction === 'in' ? 'IN' : 'OUT' }}</span>
                  </td>
                  <td class="align-right">{{ formatValue(item.value) }}</td>
                  <td class="token-cell">{{ truncateHash(item.token) }}</td>
                  <td class="align-right">{{ item.block || '—' }}</td>
                  <td class="hash">{{ truncateHash(item.hash) }}</td>
                </tr>
              </tbody>
            </table>
            <p v-else-if="!txLoading" class="empty-text" style="padding:2rem;text-align:center;">No transactions found.</p>
            <div v-if="txLoading" class="tx-loading">Loading...</div>
          </div>
          <div class="tx-footer">
            <span class="caption">{{ txItems.length }} of {{ txTotal }} transactions</span>
          </div>
        </div>
      </div>
    </Teleport>

    <section class="panel nft-panel">
      <header class="nft-header">
        <div>
          <h2>{{ t('wallet.nft_holdings') }}</h2>
          <p class="caption">{{ t('wallet.nft_caption') }}</p>
        </div>
        <div class="nft-tabs">
          <button
            class="tab"
            type="button"
            :class="{ active: nftTab === 'shown' }"
            @click="nftTab = 'shown'"
          >
            {{ t('wallet.tab_shown').replace('{count}', String(shownNfts.length)) }}
          </button>
          <button
            class="tab"
            type="button"
            :class="{ active: nftTab === 'hidden' }"
            @click="nftTab = 'hidden'"
          >
            {{ t('wallet.tab_hidden').replace('{count}', String(hiddenNfts.length)) }}
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
          {{ t('wallet.hide_selected') }}
        </button>
        <button
          v-else
          class="btn ghost"
          type="button"
          :disabled="!hiddenSelectionCount"
          @click="showSelectedNfts"
        >
          {{ t('wallet.show_selected') }}
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
            <img v-if="nft.image" :src="nft.image" :alt="nft.title || t('wallet.nft')" loading="lazy" />
            <span v-else>{{ t('wallet.no_image') }}</span>
          </div>
          <div class="nft-meta">
            <strong>{{ nft.title || t('wallet.untitled') }}</strong>
            <span class="caption">{{ nft.chain }}</span>
            <span class="token-id">#{{ nft.token_id }}</span>
          </div>
        </article>
      </div>
      <p v-else class="empty-text">
        {{ nftTab === 'shown' ? t('wallet.no_nfts_shown') : t('wallet.no_nfts_hidden') }}
      </p>
    </section>

    <section class="panel console-panel">
      <header>
        <div>
          <h2>{{ t('wallet.automation_consoles') }}</h2>
          <p class="caption">{{ t('wallet.console_caption') }}</p>
        </div>
        <button class="btn ghost" type="button" @click="refreshConsoleLogs" :disabled="consoleBusy">
          {{ consoleBusy ? t('common.refreshing') : t('wallet.refresh_logs') }}
        </button>
      </header>
      <AutomationConsoleStack :manager-lines="consoleLines" :guardian-lines="guardianConsole" />
    </section>

    <section class="panel mnemonic-panel">
      <header>
        <div>
          <h2>{{ t('wallet.seed_title') }}</h2>
          <span class="caption">{{ t('wallet.seed_caption') }}</span>
        </div>
        <span class="caption">{{ wallet.mnemonicPreview ? t('wallet.loaded') : t('wallet.not_set') }}</span>
      </header>
      <form class="mnemonic-form" @submit.prevent="saveMnemonic">
        <textarea v-model="mnemonicInput" rows="3" :placeholder="t('wallet.mnemonic_placeholder')"></textarea>
        <div class="actions">
          <button class="btn" type="submit">{{ t('common.save') }}</button>
          <button class="btn ghost" type="button" @click="clearMnemonic">{{ t('common.clear') }}</button>
        </div>
      </form>
    </section>

    <TradingStartupWizard
      v-if="wizardSteps.length"
      v-model:open="wizardOpen"
      :steps="wizardSteps"
      :title="t('wallet.wizard_title')"
      :subtitle="t('wallet.wizard_subtitle')"
      :eyebrow="t('wallet.wizard_eyebrow')"
    >
      <template #step-mnemonic>
        <div class="wizard-field">
          <label for="wizard-mnemonic">{{ t('wallet.wizard_recovery_phrase') }}</label>
          <textarea
            id="wizard-mnemonic"
            v-model="wizardMnemonicInput"
            rows="3"
            :placeholder="t('wallet.mnemonic_placeholder')"
          ></textarea>
          <div class="wizard-actions">
            <button class="btn" type="button" @click="saveWizardMnemonic" :disabled="!wizardMnemonicInput.trim()">
              {{ t('wallet.save_phrase') }}
            </button>
            <button class="btn ghost" type="button" @click="clearWizardMnemonic">
              {{ t('common.clear') }}
            </button>
          </div>
        </div>
      </template>
    </TradingStartupWizard>

    <!-- Reveal Mnemonic Modal -->
    <Teleport to="body">
      <div v-if="revealModalOpen" class="modal-overlay" @click.self="closeRevealModal">
        <div class="modal-card">
          <header class="modal-header">
            <h3>Recovery Phrase — Wallet {{ revealWalletIdx }}</h3>
            <button class="modal-close" type="button" @click="closeRevealModal">&times;</button>
          </header>
          <div class="modal-body">
            <p class="modal-warning">
              Keep this recovery phrase safe. Anyone with access to it can control this wallet.
            </p>
            <div v-if="revealLoading" class="modal-loading">Decrypting...</div>
            <div v-else-if="revealError" class="modal-error">{{ revealError }}</div>
            <div v-else class="mnemonic-display">
              <div class="mnemonic-words">
                <span v-for="(word, i) in revealWords" :key="i" class="mnemonic-word">
                  <span class="word-num">{{ i + 1 }}</span>
                  {{ word }}
                </span>
              </div>
              <button class="btn ghost" type="button" @click="copyMnemonic">
                {{ copiedMnemonic ? 'Copied' : 'Copy to Clipboard' }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue';
import { fetchWalletNftPreferences, updateWalletNftPreferences, revealWalletMnemonic, fetchWalletTransfers } from '@/api';
import type { TransferItem } from '@/api';
import { useDashboardStore } from '@/stores/dashboard';
import { useWalletStore } from '@/stores/wallet';
import AutomationConsoleStack from '@/components/AutomationConsoleStack.vue';
import TradingStartupWizard from '@/components/TradingStartupWizard.vue';
import { t } from '@/i18n';

const dashboard = useDashboardStore();
const wallet = useWalletStore();
const mnemonicInput = ref('');
const formState = reactive<Record<string, Record<string, string>>>({});
const activeAction = ref('');
const statusTimer = ref<number>();
const consoleTimer = ref<number>();
const consoleBusy = ref(false);
const creatingWallet = ref(false);
const wizardOpen = ref(!sessionStorage.getItem('wallet-wizard-dismissed'));
const wizardMnemonicInput = ref('');
const nftTab = ref<'shown' | 'hidden'>('shown');
const nftHidden = ref<Set<string>>(new Set());
const nftSelectionShown = ref<Set<string>>(new Set());
const nftSelectionHidden = ref<Set<string>>(new Set());
// Reveal mnemonic modal state
const revealModalOpen = ref(false);
const revealWalletIdx = ref(0);
const revealMnemonicValue = ref('');
const revealLoading = ref(false);
const revealError = ref('');
const copiedMnemonic = ref(false);
const revealWords = computed(() => revealMnemonicValue.value ? revealMnemonicValue.value.split(/\s+/) : []);

// All Transactions modal state
const txModalOpen = ref(false);
const txItems = ref<TransferItem[]>([]);
const txTotal = ref(0);
const txLoading = ref(false);
const txSort = ref<'asc' | 'desc'>('desc');
const txSearch = ref('');
const txOffset = ref(0);
const txScrollRef = ref<HTMLElement | null>(null);
const TX_PAGE_SIZE = 50;
let txSearchTimer: ReturnType<typeof setTimeout> | null = null;

const walletBalances = computed(() => wallet.balances);
const transferEntries = computed(() => Object.entries(wallet.transfers || {}));
const consoleLines = computed(() => dashboard.consoleLogs || []);
const guardianConsole = computed(() => dashboard.guardianLogs || []);
const snapshotTimestamp = computed(() => wallet.snapshot?.updated_at || t('common.never'));
const shownNfts = computed(() => wallet.nfts.filter((nft: any) => !nftHidden.value.has(nftKey(nft))));
const hiddenNfts = computed(() => wallet.nfts.filter((nft: any) => nftHidden.value.has(nftKey(nft))));
const activeNfts = computed(() => (nftTab.value === 'shown' ? shownNfts.value : hiddenNfts.value));
const shownSelectionCount = computed(() => nftSelectionShown.value.size);
const hiddenSelectionCount = computed(() => nftSelectionHidden.value.size);

const activeWalletCaption = computed(() => {
  if (wallet.activeWalletIndex === -1) {
    if (wallet.walletCount > 1) {
      return `${wallet.walletCount} wallets — aggregate view`;
    }
    return wallet.snapshot?.wallet || t('wallet.no_wallet');
  }
  return wallet.activeWalletAddress || t('wallet.no_wallet');
});

const workerSummary = computed(() => {
  if (wallet.running || wallet.autoRefreshing || wallet.status?.running) {
    return t('wallet.updating');
  }
  const status = wallet.status || {};
  const message = String(status.message || t('common.idle'));
  const returncode = status.returncode;
  if (message.toLowerCase().startsWith('failed') || (typeof returncode === 'number' && returncode !== 0)) {
    const finishedAt = Number(status.finished_at || 0);
    const snapshotAt = parseTimestamp(wallet.snapshot?.updated_at);
    if (snapshotAt && finishedAt && snapshotAt > (finishedAt * 1000 + 5000)) {
      return t('wallet.updated_failed');
    }
    return message;
  }
  return message === 'idle' ? t('common.idle') : message;
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

  if (!hasMnemonic) {
    steps.push({
      id: 'mnemonic',
      title: t('wallet.wizard_add_phrase_title'),
      description: t('wallet.wizard_add_phrase_desc'),
      detail: t('wallet.wizard_add_phrase_detail'),
      tone: 'critical',
    });
  }

  return steps;
});

watch(wizardOpen, (open) => {
  if (!open) sessionStorage.setItem('wallet-wizard-dismissed', '1');
});

watch(wizardSteps, (steps) => {
  if (steps.some((s) => s.tone === 'critical')) {
    sessionStorage.removeItem('wallet-wizard-dismissed');
    wizardOpen.value = true;
  }
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

async function refreshAll() {
  await wallet.fetchSnapshot();
  await wallet.fetchMultiWallet();
}

async function handleCreateWallet() {
  if (creatingWallet.value) return;
  creatingWallet.value = true;
  try {
    await wallet.createNewWallet();
  } catch (error) {
    console.warn('Failed to create wallet', error);
  } finally {
    creatingWallet.value = false;
  }
}

async function handleRevealMnemonic(walletIndex: number) {
  revealWalletIdx.value = walletIndex;
  revealMnemonicValue.value = '';
  revealError.value = '';
  copiedMnemonic.value = false;
  revealLoading.value = true;
  revealModalOpen.value = true;
  try {
    const result = await revealWalletMnemonic(walletIndex);
    revealMnemonicValue.value = result.mnemonic || '';
  } catch (error: any) {
    revealError.value = error?.response?.data?.detail || error?.message || 'Failed to reveal mnemonic';
  } finally {
    revealLoading.value = false;
  }
}

function closeRevealModal() {
  revealModalOpen.value = false;
  revealMnemonicValue.value = '';
  revealError.value = '';
  copiedMnemonic.value = false;
}

async function copyMnemonic() {
  if (!revealMnemonicValue.value) return;
  try {
    await navigator.clipboard.writeText(revealMnemonicValue.value);
    copiedMnemonic.value = true;
    setTimeout(() => { copiedMnemonic.value = false; }, 2000);
  } catch (error) {
    console.warn('Clipboard copy failed', error);
  }
}

const CHAIN_NAMES: Record<string, string> = {
  ethereum: 'Ethereum',
  polygon: 'Polygon',
  base: 'Base',
  arbitrum: 'Arbitrum',
  optimism: 'Optimism',
  avalanche: 'Avalanche',
  bsc: 'BNB Chain',
  fantom: 'Fantom',
  zksync: 'zkSync',
  linea: 'Linea',
  scroll: 'Scroll',
  blast: 'Blast',
};

function chainDisplayName(chain: string): string {
  const lower = (chain || '').toLowerCase();
  return CHAIN_NAMES[lower] || chain.charAt(0).toUpperCase() + chain.slice(1);
}

async function loadTxPage(reset = false) {
  if (txLoading.value) return;
  txLoading.value = true;
  try {
    const offset = reset ? 0 : txOffset.value;
    const result = await fetchWalletTransfers({
      offset,
      limit: TX_PAGE_SIZE,
      sort: txSort.value,
      search: txSearch.value || undefined,
    });
    if (reset) {
      txItems.value = result.items;
    } else {
      txItems.value = [...txItems.value, ...result.items];
    }
    txTotal.value = result.total;
    txOffset.value = offset + result.items.length;
  } catch (error) {
    console.warn('Failed to load transactions', error);
  } finally {
    txLoading.value = false;
  }
}

function openAllTransactions() {
  txModalOpen.value = true;
  txItems.value = [];
  txTotal.value = 0;
  txOffset.value = 0;
  txSearch.value = '';
  txSort.value = 'desc';
  loadTxPage(true);
}

function closeTxModal() {
  txModalOpen.value = false;
  txItems.value = [];
}

function toggleTxSort() {
  txSort.value = txSort.value === 'desc' ? 'asc' : 'desc';
  loadTxPage(true);
}

function debouncedTxSearch() {
  if (txSearchTimer) clearTimeout(txSearchTimer);
  txSearchTimer = setTimeout(() => {
    loadTxPage(true);
  }, 350);
}

function handleTxScroll() {
  const el = txScrollRef.value;
  if (!el || txLoading.value) return;
  if (txItems.value.length >= txTotal.value) return;
  const threshold = 100;
  if (el.scrollHeight - el.scrollTop - el.clientHeight < threshold) {
    loadTxPage(false);
  }
}

async function refreshConsoleLogs() {
  if (consoleBusy.value) return;
  consoleBusy.value = true;
  try {
    await dashboard.refreshConsole();
  } catch (error) {
    console.warn('Failed to refresh console logs', error);
  } finally {
    consoleBusy.value = false;
  }
}

onMounted(async () => {
  await Promise.all([
    wallet.refreshStatus(),
    wallet.fetchSnapshot(),
    wallet.fetchMultiWallet(),
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
      wallet.fetchMultiWallet();
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
  if (!value) return t('common.none');
  if (typeof value === 'string' && value.startsWith('0x')) {
    return parseInt(value, 16).toString();
  }
  return value;
}

function truncateHash(hash: string | undefined) {
  if (!hash) return t('common.unknown');
  return `${hash.slice(0, 6)}…${hash.slice(-4)}`;
}

function truncateAddress(addr: string | null | undefined) {
  if (!addr) return '—';
  return `${addr.slice(0, 6)}…${addr.slice(-4)}`;
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
    const hiddenKeys = new Set<string>((data.items || []).map((item: any) => nftKey(item)));
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

/* ── Wallet Tab Bar ────────────────────────────────── */
.wallet-tabs-panel {
  padding: 0.8rem 1.2rem;
}

.wallet-tab-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.wallet-tab {
  border-radius: 999px;
  border: 1px solid rgba(111, 167, 255, 0.3);
  background: transparent;
  color: #f4f6fa;
  padding: 0.5rem 1.2rem;
  cursor: pointer;
  font-size: 0.88rem;
  transition: background 0.15s, border-color 0.15s;
}

.wallet-tab:hover {
  background: rgba(45, 117, 196, 0.18);
}

.wallet-tab.active {
  background: rgba(45, 117, 196, 0.35);
  border-color: rgba(45, 117, 196, 0.65);
  font-weight: 600;
}

.wallet-tab.create-tab {
  border-color: rgba(52, 211, 153, 0.4);
  color: #34d399;
}

.wallet-tab.create-tab:hover {
  background: rgba(52, 211, 153, 0.12);
}

/* ── Summary ────────────────────────────────────────── */
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

.summary-grid .value.highlight {
  color: #34d399;
  font-size: 1.4rem;
}

/* ── Wallet Breakdown (All Wallets view) ───────────── */
.wallet-breakdown {
  margin-top: 1rem;
  border-top: 1px solid rgba(111, 167, 255, 0.12);
  padding-top: 0.8rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.breakdown-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.3rem 0;
}

.breakdown-label {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.breakdown-addr {
  font-size: 0.78rem;
  color: rgba(255, 255, 255, 0.5);
  font-family: monospace;
}

.breakdown-value {
  font-weight: 600;
  font-size: 1rem;
}

.link-btn {
  background: none;
  border: none;
  color: #6fa7ff;
  cursor: pointer;
  padding: 0;
  font-size: 0.88rem;
  text-decoration: underline;
  text-decoration-color: rgba(111, 167, 255, 0.3);
}

.link-btn:hover {
  color: #93c5fd;
}

/* ── Config Panel ───────────────────────────────────── */
.config-panel {
  padding: 0.8rem 1.2rem;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.8rem;
}

.config-grid .label {
  display: block;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08rem;
  color: rgba(255, 255, 255, 0.55);
}

.config-grid .value {
  font-size: 0.95rem;
  font-weight: 500;
}

/* ── Actions ────────────────────────────────────────── */
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

/* ── Reveal Row ────────────────────────────────────── */
.breakdown-actions {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.btn-tiny {
  font-size: 0.72rem;
  padding: 0.2rem 0.6rem;
  border-radius: 6px;
  border: 1px solid rgba(111, 167, 255, 0.3);
  background: transparent;
  color: rgba(255, 255, 255, 0.6);
  cursor: pointer;
}

.btn-tiny:hover {
  color: #f4f6fa;
  border-color: rgba(111, 167, 255, 0.5);
}

.reveal-row {
  margin-top: 0.8rem;
  padding-top: 0.6rem;
  border-top: 1px solid rgba(111, 167, 255, 0.1);
}

/* ── Reveal Mnemonic Modal ─────────────────────────── */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  backdrop-filter: blur(4px);
}

.modal-card {
  background: #0e1a2d;
  border: 1px solid rgba(111, 167, 255, 0.25);
  border-radius: 20px;
  max-width: 560px;
  width: 90%;
  padding: 0;
  overflow: hidden;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.2rem 1.5rem;
  border-bottom: 1px solid rgba(111, 167, 255, 0.12);
  margin-bottom: 0;
}

.modal-header h3 {
  margin: 0;
  font-size: 1.1rem;
  color: #f4f6fa;
}

.modal-close {
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.6);
  font-size: 1.6rem;
  cursor: pointer;
  padding: 0;
  line-height: 1;
}

.modal-close:hover {
  color: #f87171;
}

.modal-body {
  padding: 1.2rem 1.5rem 1.5rem;
}

.modal-warning {
  color: #fbbf24;
  font-size: 0.85rem;
  margin-bottom: 1rem;
  padding: 0.6rem 0.8rem;
  background: rgba(251, 191, 36, 0.08);
  border: 1px solid rgba(251, 191, 36, 0.2);
  border-radius: 10px;
}

.modal-loading {
  text-align: center;
  color: rgba(255, 255, 255, 0.6);
  padding: 2rem 0;
}

.modal-error {
  color: #f87171;
  text-align: center;
  padding: 1rem 0;
}

.mnemonic-display {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.mnemonic-words {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.5rem;
}

.mnemonic-word {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.7rem;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(111, 167, 255, 0.15);
  border-radius: 8px;
  font-family: monospace;
  font-size: 0.9rem;
  color: #f4f6fa;
}

.word-num {
  font-size: 0.7rem;
  color: rgba(255, 255, 255, 0.35);
  min-width: 1.2rem;
}

/* ── Transfer card enhancements ──────────────────────── */
.transfer-card header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.6rem;
}

.transfer-count {
  font-size: 0.72rem;
  background: rgba(111, 167, 255, 0.15);
  color: rgba(255, 255, 255, 0.7);
  border-radius: 999px;
  padding: 0.15rem 0.55rem;
}

.more-link {
  text-align: center;
  padding-top: 0.3rem;
}

/* ── All Transactions Modal ──────────────────────────── */
.tx-modal {
  max-width: 900px;
  width: 95%;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
}

.tx-toolbar {
  display: flex;
  gap: 0.6rem;
  padding: 0.8rem 1.5rem;
  border-bottom: 1px solid rgba(111, 167, 255, 0.1);
  align-items: center;
}

.tx-search-input {
  flex: 1;
  border-radius: 10px;
  border: 1px solid rgba(111, 167, 255, 0.25);
  background: rgba(255, 255, 255, 0.04);
  color: #f4f6fa;
  padding: 0.5rem 0.8rem;
  font-size: 0.88rem;
}

.tx-search-input::placeholder {
  color: rgba(255, 255, 255, 0.35);
}

.tx-scroll {
  flex: 1;
  overflow-y: auto;
  min-height: 200px;
  max-height: calc(85vh - 180px);
}

.tx-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.tx-table thead {
  position: sticky;
  top: 0;
  background: #0e1a2d;
  z-index: 1;
}

.tx-table th {
  text-align: left;
  padding: 0.6rem 0.8rem;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08rem;
  color: rgba(255, 255, 255, 0.5);
  border-bottom: 1px solid rgba(111, 167, 255, 0.12);
}

.tx-table td {
  padding: 0.5rem 0.8rem;
  border-bottom: 1px solid rgba(111, 167, 255, 0.06);
  color: #f4f6fa;
}

.tx-table tbody tr:hover {
  background: rgba(45, 117, 196, 0.08);
}

.token-cell {
  font-family: monospace;
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.6);
}

.tx-loading {
  text-align: center;
  color: rgba(255, 255, 255, 0.5);
  padding: 1.5rem;
}

.tx-footer {
  padding: 0.6rem 1.5rem;
  border-top: 1px solid rgba(111, 167, 255, 0.1);
  display: flex;
  justify-content: flex-end;
}
</style>
