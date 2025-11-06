import { defineStore } from 'pinia';
import { fetchLabFiles, fetchLabStatus, startLabJob, fetchLabNews, LabJobPayload } from '@/api';

interface LabState {
  files: Record<string, any>[];
  status: Record<string, any> | null;
  loading: boolean;
  error: string | null;
  news: Record<string, any>[];
  newsMeta: Record<string, any> | null;
  newsLoading: boolean;
  newsError: string | null;
}

export const useLabStore = defineStore('lab', {
  state: (): LabState => ({
    files: [],
    status: null,
    loading: false,
    error: null,
    news: [],
    newsMeta: null,
    newsLoading: false,
    newsError: null,
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
  },
});
