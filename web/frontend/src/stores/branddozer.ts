import { defineStore } from 'pinia';
import {
  fetchBrandProjects,
  createBrandProject,
  updateBrandProject,
  deleteBrandProject,
  startBrandProject,
  stopBrandProject,
  fetchBrandLogs,
  generateBrandInterjections,
  fetchBrandRoots,
  importBrandProjectFromGitHub,
} from '@/api';

interface BrandProject {
  id: string;
  name: string;
  root_path: string;
  default_prompt: string;
  interjections: string[];
  interval_minutes: number;
  enabled?: boolean;
  running?: boolean;
  state?: string;
  last_run?: number;
  last_ai_generated?: number;
  last_message?: string;
  repo_url?: string;
  repo_branch?: string;
  created_at?: string;
  updated_at?: string;
}

interface BrandDozerState {
  projects: BrandProject[];
  logs: string[];
  loading: boolean;
  saving: boolean;
  logLoading: boolean;
  importing: boolean;
  error: string | null;
}

export const useBrandDozerStore = defineStore('branddozer', {
  state: (): BrandDozerState => ({
    projects: [],
    logs: [],
    loading: false,
    saving: false,
    logLoading: false,
    importing: false,
    error: null,
  }),
  actions: {
    async load() {
      this.loading = true;
      try {
        const data = await fetchBrandProjects();
        this.projects = data.projects || [];
        this.error = null;
      } catch (err: any) {
        this.error = err?.message || 'Failed to load projects';
      } finally {
        this.loading = false;
      }
    },
    async create(payload: Partial<BrandProject>) {
      this.saving = true;
      try {
        await createBrandProject(payload);
        await this.load();
      } finally {
        this.saving = false;
      }
    },
    async update(id: string, payload: Partial<BrandProject>) {
      this.saving = true;
      try {
        await updateBrandProject(id, payload);
        await this.load();
      } finally {
        this.saving = false;
      }
    },
    async remove(id: string) {
      await deleteBrandProject(id);
      await this.load();
    },
    async start(id: string) {
      await startBrandProject(id);
      await this.load();
    },
    async stop(id: string) {
      await stopBrandProject(id);
      await this.load();
    },
    async refreshLogs(id: string, limit = 200) {
      this.logLoading = true;
      try {
        const data = await fetchBrandLogs(id, limit);
        this.logs = data.lines || [];
      } finally {
        this.logLoading = false;
      }
    },
    async generateInterjections(id: string, defaultPrompt?: string) {
      this.saving = true;
      try {
        const data = await generateBrandInterjections(id, defaultPrompt);
        return data.interjections || [];
      } finally {
        this.saving = false;
      }
    },
    async browseRoots(path?: string) {
      const data = await fetchBrandRoots(path);
      return data;
    },
    async importFromGitHub(payload: Record<string, any>) {
      this.importing = true;
      try {
        const data = await importBrandProjectFromGitHub(payload);
        await this.load();
        return data.project;
      } finally {
        this.importing = false;
      }
    },
  },
});
