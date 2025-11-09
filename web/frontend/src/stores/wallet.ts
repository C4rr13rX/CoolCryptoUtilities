import { defineStore } from 'pinia';
import {
  fetchWalletStatus,
  runWalletAction,
  fetchWalletMnemonic,
  updateWalletMnemonic,
  fetchWalletState,
} from '@/api';

interface WalletState {
  status: Record<string, any> | null;
  loading: boolean;
  error: string | null;
  mnemonicPreview: string;
  running: boolean;
  snapshot: Record<string, any> | null;
  snapshotLoading: boolean;
  autoRefreshing: boolean;
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
  }),
  getters: {
    actions(state) {
      return state.status?.actions || [];
    },
    currentLog(state) {
      return state.status?.log || [];
    },
    balances(state) {
      return state.snapshot?.balances || [];
    },
    transfers(state) {
      return state.snapshot?.transfers || {};
    },
    nfts(state) {
      return state.snapshot?.nfts || [];
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
        await this.run('refresh_transfers');
        await this.run('refresh_balances');
        this.error = null;
      } catch (error) {
        // error already captured by run; swallow to avoid loop
      } finally {
        this.autoRefreshing = false;
      }
    },
  },
});
