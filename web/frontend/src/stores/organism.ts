import { defineStore } from 'pinia';
import {
  fetchOrganismConfig,
  fetchOrganismHistory,
  fetchOrganismLatest,
  updateOrganismConfig,
  OrganismHistoryParams,
} from '@/api';

interface OrganismState {
  latest: Record<string, any> | null;
  history: Record<string, any>[];
  loading: boolean;
  error: string | null;
  lastUpdated: number | null;
  labelScale: number;
}

function normaliseSnapshot(snapshot: Record<string, any> | null) {
  if (!snapshot) return null;
  const ts = Number(snapshot.timestamp ?? snapshot.ts ?? Date.now() / 1000);
  return { ...snapshot, timestamp: ts };
}

export const useOrganismStore = defineStore('organism', {
  state: (): OrganismState => ({
    latest: null,
    history: [],
    loading: false,
    error: null,
    lastUpdated: null,
    labelScale: 1,
  }),
  getters: {
    timeline(state): number[] {
      const points = state.history.map((entry) => Number(entry.timestamp || 0)).filter((val) => !Number.isNaN(val));
      if (state.latest?.timestamp) {
        points.unshift(Number(state.latest.timestamp));
      }
      return Array.from(new Set(points)).sort((a, b) => a - b);
    },
  },
  actions: {
    appendSnapshot(snapshot: Record<string, any>) {
      const normalised = normaliseSnapshot(snapshot);
      if (!normalised || !normalised.timestamp) {
        return;
      }
      this.latest = normalised;
      this.lastUpdated = normalised.timestamp;
      const existingIndex = this.history.findIndex(
        (entry) => Number(entry.timestamp) === Number(normalised.timestamp),
      );
      if (existingIndex >= 0) {
        this.history.splice(existingIndex, 1, normalised);
      } else {
        this.history.unshift(normalised);
        this.history.sort((a, b) => Number(b.timestamp) - Number(a.timestamp));
        if (this.history.length > 512) {
          this.history.length = 512;
        }
      }
    },
    async refreshLatest() {
      this.loading = true;
      try {
        const response = await fetchOrganismLatest();
        const snapshot = response?.snapshot || null;
        if (snapshot) {
          this.appendSnapshot(snapshot);
        }
        this.error = null;
      } catch (error: any) {
        this.error = error?.message || 'Failed to load organism snapshot';
      } finally {
        this.loading = false;
      }
    },
    async loadHistory(params?: OrganismHistoryParams) {
      this.loading = true;
      try {
        const response = await fetchOrganismHistory(params);
        const entries: Record<string, any>[] = response?.snapshots || [];
        this.history = entries
          .map((entry) => normaliseSnapshot(entry))
          .filter((entry): entry is Record<string, any> => Boolean(entry))
          .sort((a, b) => Number(b.timestamp) - Number(a.timestamp));
        if (this.latest) {
          this.appendSnapshot(this.latest);
        }
        this.error = null;
      } catch (error: any) {
        this.error = error?.message || 'Failed to load organism history';
      } finally {
        this.loading = false;
      }
    },
    async loadSettings() {
      try {
        const response = await fetchOrganismConfig();
        const scale = Number(response?.label_scale ?? 1);
        if (!Number.isNaN(scale)) {
          this.labelScale = scale;
        }
      } catch (error: any) {
        this.error = error?.message || 'Failed to load organism settings';
      }
    },
    setLabelScaleLocal(scale: number) {
      const safe = Number.isFinite(scale) ? scale : 1;
      this.labelScale = safe;
    },
    async saveLabelScale(scale: number) {
      const safe = Number.isFinite(scale) ? scale : 1;
      this.labelScale = safe;
      try {
        await updateOrganismConfig({ label_scale: safe });
      } catch (error: any) {
        this.error = error?.message || 'Failed to update label scale';
      }
    },
  },
});
