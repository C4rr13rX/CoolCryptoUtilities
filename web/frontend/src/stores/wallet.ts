import { defineStore } from 'pinia';
import {
  fetchWalletStatus,
  runWalletAction,
  fetchWalletMnemonic,
  updateWalletMnemonic,
  fetchWalletState,
  fetchMultiWalletState,
  createMultiWallet,
} from '@/api';
import type { MultiWalletState, MultiWalletSummary, MultiWalletConfig } from '@/api';

interface WalletState {
  status: Record<string, any> | null;
  loading: boolean;
  error: string | null;
  mnemonicPreview: string;
  running: boolean;
  snapshot: Record<string, any> | null;
  snapshotLoading: boolean;
  autoRefreshing: boolean;
  // Multi-wallet state
  multiWallet: MultiWalletState | null;
  multiWalletLoading: boolean;
  activeWalletIndex: number; // -1 = aggregate "All Wallets" view
}

export const useWalletStore = defineStore('wallet', {
  state: (): WalletState => ({
    status: null,
    loading: false,
    error: null,
    mnemonicPreview: '',
    running: false,
    snapshot: null,
    snapshotLoading: false,
    autoRefreshing: false,
    multiWallet: null,
    multiWalletLoading: false,
    activeWalletIndex: -1,
  }),
  getters: {
    actions(state) {
      return state.status?.actions || [];
    },
    currentLog(state) {
      return state.status?.log || [];
    },
    // Aggregate or per-wallet balances depending on active tab
    balances(state): Record<string, any>[] {
      if (!state.multiWallet) return state.snapshot?.balances || [];
      if (state.activeWalletIndex === -1) return state.multiWallet.balances || [];
      return (state.multiWallet.balances || []).filter(
        (b: any) => b.wallet_index === state.activeWalletIndex,
      );
    },
    transfers(state): Record<string, any> {
      if (!state.multiWallet) return state.snapshot?.transfers || {};
      return state.multiWallet.transfers || {};
    },
    nfts(state): Record<string, any>[] {
      if (!state.multiWallet) return state.snapshot?.nfts || [];
      if (state.activeWalletIndex === -1) return state.multiWallet.nfts || [];
      return (state.multiWallet.nfts || []).filter(
        (n: any) => n.wallet_index === state.activeWalletIndex,
      );
    },
    walletList(state): MultiWalletSummary[] {
      return state.multiWallet?.wallets || [];
    },
    walletCount(state): number {
      return state.multiWallet?.wallet_count || 0;
    },
    totalUsd(state): number {
      if (state.multiWallet) return state.multiWallet.totals?.usd || 0;
      return state.snapshot?.totals?.usd || 0;
    },
    multiWalletConfig(state): MultiWalletConfig | null {
      return state.multiWallet?.config || null;
    },
    activeWalletAddress(state): string | null {
      if (!state.multiWallet || state.activeWalletIndex === -1) return null;
      const w = state.multiWallet.wallets.find((entry: MultiWalletSummary) => entry.index === state.activeWalletIndex);
      return w?.wallet || null;
    },
    activeWalletUsd(state): number {
      if (!state.multiWallet || state.activeWalletIndex === -1) {
        return state.multiWallet?.totals?.usd || state.snapshot?.totals?.usd || 0;
      }
      const w = state.multiWallet.wallets.find((entry: MultiWalletSummary) => entry.index === state.activeWalletIndex);
      return w?.usd || 0;
    },
  },
  actions: {
    async refreshStatus() {
      this.loading = true;
      try {
        const data = await fetchWalletStatus();
        this.status = data;
        this.running = Boolean(data?.running);
        this.error = null;
      } catch (error: any) {
        this.error = error?.message || 'Failed to load wallet state';
      } finally {
        this.loading = false;
      }
    },
    async fetchSnapshot() {
      this.snapshotLoading = true;
      try {
        const data = await fetchWalletState();
        this.snapshot = data || null;
        this.error = null;
      } catch (error: any) {
        this.error = error?.message || 'Failed to load wallet snapshot';
      } finally {
        this.snapshotLoading = false;
      }
    },
    async fetchMultiWallet() {
      this.multiWalletLoading = true;
      try {
        const data = await fetchMultiWalletState();
        this.multiWallet = data || null;
        this.error = null;
      } catch (error: any) {
        this.error = error?.message || 'Failed to load multi-wallet state';
      } finally {
        this.multiWalletLoading = false;
      }
    },
    async createNewWallet() {
      try {
        const result = await createMultiWallet();
        await this.fetchMultiWallet();
        return result;
      } catch (error: any) {
        this.error = error?.response?.data?.detail || error?.message || 'Failed to create wallet';
        throw error;
      }
    },
    setActiveWallet(index: number) {
      this.activeWalletIndex = index;
    },
    async run(action: string, options?: Record<string, any>) {
      if (this.running) return;
      this.running = true;
      try {
        await runWalletAction({ action, options });
        await this._pollUntilIdle();
        this.error = null;
      } catch (error: any) {
        this.error = error?.response?.data?.detail || error?.message || 'Wallet action failed';
        throw error;
      } finally {
        this.running = false;
      }
    },
    async _pollUntilIdle(timeoutMs = 120000, intervalMs = 2500) {
      const start = Date.now();
      while (Date.now() - start < timeoutMs) {
        await this.refreshStatus();
        if (!this.status?.running) {
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
      }
      await this.fetchSnapshot();
      await this.fetchMultiWallet();
    },
    async loadMnemonicPreview() {
      const data = await fetchWalletMnemonic();
      this.mnemonicPreview = data?.preview || '';
    },
    async saveMnemonic(mnemonic: string | null) {
      await updateWalletMnemonic(mnemonic);
      await this.loadMnemonicPreview();
    },
    async autoRefresh() {
      if (this.autoRefreshing) return;
      this.autoRefreshing = true;
      try {
        let hadError = false;
        try {
          await this.run('refresh_transfers');
        } catch (error) {
          // Continue to balances even if transfers failed.
          hadError = true;
        }
        try {
          await this.run('refresh_balances');
        } catch (error) {
          // Keep the most recent error, but do not abort auto refresh.
          hadError = true;
        }
        if (!hadError) {
          this.error = null;
        }
      } catch (error) {
        // error already captured by run; swallow to avoid loop
      } finally {
        this.autoRefreshing = false;
      }
    },
  },
});
