import { defineStore } from 'pinia';
import {
  fetchDataLabDatasets,
  startDataLabJob,
  fetchDataLabJobStatus,
  fetchDataLabNews,
  fetchDataLabSignals,
  fetchDataLabWatchlists,
  updateDataLabWatchlist,
  DataLabDatasetParams,
  DataLabJobPayload,
  DataLabNewsPayload,
  DataLabSignalParams,
  DataLabWatchlistUpdatePayload,
} from '@/api';

interface DataLabState {
  datasets: Record<string, any>[];
  datasetsLoading: boolean;
  datasetsError: string | null;
  jobStatus: Record<string, any> | null;
  jobLoading: boolean;
  jobError: string | null;
  jobHistory: Record<string, any>[];
  newsItems: Record<string, any>[];
  newsMeta: Record<string, any> | null;
  newsLoading: boolean;
  newsError: string | null;
  signals: Record<string, any>[];
  signalsLoading: boolean;
  signalsError: string | null;
  signalsMeta: Record<string, any> | null;
  watchlists: Record<string, string[]>;
  watchlistsLoading: boolean;
  watchlistsError: string | null;
  lastActionMessage: string | null;
}

export const useDataLabStore = defineStore('dataLab', {
  state: (): DataLabState => ({
    datasets: [],
    datasetsLoading: false,
    datasetsError: null,
    jobStatus: null,
    jobLoading: false,
    jobError: null,
    jobHistory: [],
    newsItems: [],
    newsMeta: null,
    newsLoading: false,
    newsError: null,
    signals: [],
    signalsLoading: false,
    signalsError: null,
    signalsMeta: null,
    watchlists: { stream: [], ghost: [], live: [] },
    watchlistsLoading: false,
    watchlistsError: null,
    lastActionMessage: null,
  }),
  getters: {
    runningJob(state): boolean {
      return Boolean(state.jobStatus?.running);
    },
    jobLog(state): string[] {
      return (state.jobStatus?.log as string[]) || [];
    },
    jobHistory(state): Record<string, any>[] {
      return state.jobHistory;
    },
  },
  actions: {
    async loadDatasets(params?: DataLabDatasetParams) {
      this.datasetsLoading = true;
      try {
        const data = await fetchDataLabDatasets(params);
        this.datasets = Array.isArray(data?.items) ? data.items : [];
        this.datasetsError = null;
      } catch (err: any) {
        this.datasetsError = err?.message || 'Failed to load datasets';
        this.datasets = [];
      } finally {
        this.datasetsLoading = false;
      }
    },
    async runJob(payload: DataLabJobPayload) {
      this.jobLoading = true;
      try {
        const status = await startDataLabJob(payload);
        this.jobStatus = status;
        this.jobHistory = Array.isArray(status?.history) ? status.history : [];
        this.jobError = null;
      } catch (err: any) {
        this.jobError = err?.message || 'Failed to start data lab job';
        throw err;
      } finally {
        this.jobLoading = false;
      }
    },
    async refreshJobStatus() {
      try {
        const status = await fetchDataLabJobStatus();
        this.jobStatus = status;
        this.jobHistory = Array.isArray(status?.history) ? status.history : this.jobHistory;
      } catch (err: any) {
        this.jobError = err?.message || 'Failed to fetch job status';
      }
    },
    async loadNews(payload: DataLabNewsPayload) {
      this.newsLoading = true;
      try {
        const data = await fetchDataLabNews(payload);
        this.newsItems = Array.isArray(data?.items) ? data.items : [];
        this.newsMeta = data || null;
        this.newsError = null;
      } catch (err: any) {
        this.newsError = err?.message || 'Failed to fetch news';
        this.newsItems = [];
        this.newsMeta = null;
      } finally {
        this.newsLoading = false;
      }
    },
    async loadSignals(params?: DataLabSignalParams) {
      this.signalsLoading = true;
      try {
        const data = await fetchDataLabSignals(params);
        this.signals = Array.isArray(data?.items) ? data.items : [];
        this.signalsMeta = data?.meta || null;
        if (data?.watchlists) {
          this.watchlists = data.watchlists;
        }
        this.signalsError = null;
      } catch (err: any) {
        this.signalsError = err?.message || 'Failed to fetch price signals';
        this.signals = [];
        this.signalsMeta = null;
      } finally {
        this.signalsLoading = false;
      }
    },
    async refreshWatchlists() {
      this.watchlistsLoading = true;
      try {
        const data = await fetchDataLabWatchlists();
        this.watchlists = data?.watchlists || { stream: [], ghost: [], live: [] };
        this.watchlistsError = null;
      } catch (err: any) {
        this.watchlistsError = err?.message || 'Failed to load watchlists';
      } finally {
        this.watchlistsLoading = false;
      }
    },
    async updateWatchlist(target: 'stream' | 'ghost' | 'live', action: 'add' | 'remove' | 'set', symbols: string[]) {
      if (!symbols?.length) return;
      const payload: DataLabWatchlistUpdatePayload = { target, action, symbols };
      try {
        const data = await updateDataLabWatchlist(payload);
        this.watchlists = data?.watchlists || this.watchlists;
        this.lastActionMessage = `Watchlist ${target} updated (${action}).`;
      } catch (err: any) {
        this.watchlistsError = err?.message || 'Failed to update watchlist';
        throw err;
      }
    },
  },
});
