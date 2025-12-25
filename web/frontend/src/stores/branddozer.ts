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
  fetchBrandGithubAccount,
  saveBrandGithubAccount,
  fetchBrandGithubRepos,
  fetchBrandGithubBranches,
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
  githubConnected: boolean;
  githubHasToken: boolean;
  githubUsername: string;
  githubProfile: Record<string, any> | null;
  githubRepos: any[];
  githubBranches: any[];
  githubAccountLoading: boolean;
  githubRepoLoading: boolean;
  githubBranchLoading: boolean;
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
    githubConnected: false,
    githubHasToken: false,
    githubUsername: '',
    githubProfile: null,
    githubRepos: [],
    githubBranches: [],
    githubAccountLoading: false,
    githubRepoLoading: false,
    githubBranchLoading: false,
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
    async loadGithubAccount() {
      this.githubAccountLoading = true;
      try {
        const data = await fetchBrandGithubAccount();
        this.githubConnected = Boolean(data.connected);
        this.githubHasToken = Boolean(data.has_token ?? data.hasToken ?? data.connected);
        this.githubUsername = data.username || '';
        this.githubProfile = data.profile || null;
        this.error = null;
        return data;
      } catch (err: any) {
        this.error = err?.message || 'Failed to load GitHub account';
        throw err;
      } finally {
        this.githubAccountLoading = false;
      }
    },
    async saveGithubAccount(payload: { username?: string; token: string }) {
      this.githubAccountLoading = true;
      try {
        const data = await saveBrandGithubAccount(payload);
        this.githubConnected = Boolean(data.connected);
        this.githubHasToken = Boolean(data.has_token ?? data.hasToken ?? true);
        this.githubUsername = data.username || payload.username || this.githubUsername;
        this.githubProfile = data.profile || null;
        this.error = null;
        return data;
      } finally {
        this.githubAccountLoading = false;
      }
    },
    async fetchGithubRepos(username?: string) {
      this.githubRepoLoading = true;
      try {
        const data = await fetchBrandGithubRepos(username);
        this.githubRepos = data.repos || [];
        this.githubConnected = true;
        if (data.username) {
          this.githubUsername = data.username;
        }
        if (data.count) {
          this.githubHasToken = true;
        }
        return this.githubRepos;
      } finally {
        this.githubRepoLoading = false;
      }
    },
    async fetchGithubBranches(repoFullName: string) {
      this.githubBranchLoading = true;
      try {
        const data = await fetchBrandGithubBranches(repoFullName);
        this.githubBranches = data.branches || [];
        this.githubConnected = true;
        return this.githubBranches;
      } finally {
        this.githubBranchLoading = false;
      }
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
