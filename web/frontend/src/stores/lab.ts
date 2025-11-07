import { defineStore } from 'pinia';
import {
  fetchLabFiles,
  fetchLabStatus,
  startLabJob,
  fetchLabNews,
  fetchLabPreview,
  LabJobPayload,
  LabPreviewPayload,
} from '@/api';

interface LabState {
  files: Record<string, any>[];
  status: Record<string, any> | null;
  loading: boolean;
  error: string | null;
  history: Record<string, any>[];
  log: string[];
  news: Record<string, any>[];
  newsMeta: Record<string, any> | null;
  newsLoading: boolean;
  newsError: string | null;
  preview: Record<string, any> | null;
  previewLoading: boolean;
  previewError: string | null;
}

export const useLabStore = defineStore('lab', {
  state: (): LabState => ({
    files: [],
    status: null,
    loading: false,
    error: null,
    history: [],
    log: [],
    news: [],
    newsMeta: null,
    newsLoading: false,
    newsError: null,
    preview: null,
    previewLoading: false,
    previewError: null,
  }),
  getters: {
    running(state): boolean {
      return Boolean(state.status?.running);
    },
    progress(state): number {
      return Number(state.status?.progress ?? 0);
    },
    message(state): string {
      return String(state.status?.message ?? '');
    },
    result(state): Record<string, any> | null {
      return (state.status?.result as Record<string, any>) || null;
    },
    jobLog(state): string[] {
      return state.log;
    },
    jobHistory(state): Record<string, any>[] {
      return state.history;
    },
    events(state): Record<string, any>[] {
      return Array.isArray(state.status?.events) ? state.status?.events ?? [] : [];
    },
    snapshot(state): Record<string, any> | null {
      return (state.status?.snapshot as Record<string, any>) || null;
    },
  },
  actions: {
    async loadFiles() {
      this.loading = true;
      try {
        const data = await fetchLabFiles();
        this.files = Array.isArray(data?.files) ? data.files : [];
        this.error = null;
      } catch (err: any) {
        this.error = err?.message || 'Failed to load lab files';
      } finally {
        this.loading = false;
      }
    },
    async refreshStatus() {
      try {
        const data = await fetchLabStatus();
        this.status = data || {};
        this.log = Array.isArray(data?.log) ? data.log : [];
        this.history = Array.isArray(data?.history) ? data.history : [];
        this.error = null;
      } catch (err: any) {
        this.error = err?.message || 'Failed to fetch lab status';
      }
    },
    async runJob(payload: LabJobPayload) {
      this.loading = true;
      try {
        const data = await startLabJob(payload);
        this.status = data || {};
        this.log = Array.isArray(data?.log) ? data.log : [];
        this.history = Array.isArray(data?.history) ? data.history : [];
        this.error = null;
      } catch (err: any) {
        this.error = err?.message || 'Failed to start lab job';
        throw err;
      } finally {
        this.loading = false;
      }
    },
    async loadNews(payload: { train_files?: string[]; eval_files?: string[] }) {
      this.newsLoading = true;
      try {
        const data = await fetchLabNews(payload);
        this.news = Array.isArray(data?.items) ? data.items : [];
        this.newsMeta = data || {};
        this.newsError = null;
      } catch (err: any) {
        this.newsError = err?.message || 'Failed to fetch lab news';
        this.news = [];
        this.newsMeta = null;
      } finally {
        this.newsLoading = false;
      }
    },
    async loadPreview(payload: LabPreviewPayload) {
      this.previewLoading = true;
      try {
        const data = await fetchLabPreview(payload);
        this.preview = data || {};
        this.previewError = null;
      } catch (err: any) {
        this.previewError = err?.message || 'Failed to generate preview';
        this.preview = null;
        throw err;
      } finally {
        this.previewLoading = false;
      }
    },
    resetPreview() {
      this.preview = null;
      this.previewError = null;
    },
  },
});
