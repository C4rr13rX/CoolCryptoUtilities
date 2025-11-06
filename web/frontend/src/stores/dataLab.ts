import { defineStore } from 'pinia';
import {
  fetchDataLabDatasets,
  startDataLabJob,
  fetchDataLabJobStatus,
  fetchDataLabNews,
  DataLabDatasetParams,
  DataLabJobPayload,
  DataLabNewsPayload,
} from '@/api';

interface DataLabState {
  datasets: Record<string, any>[];
  datasetsLoading: boolean;
  datasetsError: string | null;
  jobStatus: Record<string, any> | null;
  jobLoading: boolean;
  jobError: string | null;
  newsItems: Record<string, any>[];
  newsMeta: Record<string, any> | null;
  newsLoading: boolean;
  newsError: string | null;
}

export const useDataLabStore = defineStore('dataLab', {
  state: (): DataLabState => ({
    datasets: [],
    datasetsLoading: false,
    datasetsError: null,
    jobStatus: null,
    jobLoading: false,
    jobError: null,
    newsItems: [],
    newsMeta: null,
    newsLoading: false,
    newsError: null,
  }),
  getters: {
    runningJob(state): boolean {
      return Boolean(state.jobStatus?.running);
    },
    jobLog(state): string[] {
      return (state.jobStatus?.log as string[]) || [];
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
  },
});
