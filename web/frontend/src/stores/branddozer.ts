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
  fetchBrandGithubImportStatus,
  publishBrandProject,
  fetchBrandGithubPublishStatus,
  setBrandGithubActiveAccount,
  previewBrandInterjections,
  startBrandDeliveryRun,
  fetchBrandDeliveryRuns,
  fetchBrandDeliveryRun,
  fetchBrandDeliveryBacklog,
  fetchBrandDeliveryGates,
  fetchBrandDeliverySessions,
  fetchBrandDeliverySessionLogs,
  fetchBrandDeliveryArtifacts,
  triggerBrandDeliveryUiCapture,
  fetchBrandDeliveryGovernance,
  fetchBrandDeliverySprints,
  acceptBrandDeliveryRun,
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
  githubAccounts: any[];
  githubActiveAccountId: string;
  githubActiveAccount: any | null;
  githubRepos: any[];
  githubBranches: any[];
  githubAccountLoading: boolean;
  githubRepoLoading: boolean;
  githubBranchLoading: boolean;
  deliveryRuns: any[];
  activeDeliveryRun: any | null;
  deliveryBacklog: any[];
  deliveryGates: any[];
  deliverySessions: any[];
  deliverySessionLogs: Record<string, string[]>;
  deliveryArtifacts: any[];
  uiCaptureLoading: boolean;
  deliveryGovernance: any[];
  deliverySprints: any[];
  deliveryLoading: boolean;
  publishing: boolean;
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
    githubAccounts: [],
    githubActiveAccountId: '',
    githubActiveAccount: null,
    githubRepos: [],
    githubBranches: [],
    githubAccountLoading: false,
    githubRepoLoading: false,
    githubBranchLoading: false,
    deliveryRuns: [],
    activeDeliveryRun: null,
    deliveryBacklog: [],
    deliveryGates: [],
    deliverySessions: [],
    deliverySessionLogs: {},
    deliveryArtifacts: [],
    uiCaptureLoading: false,
    deliveryGovernance: [],
    deliverySprints: [],
    deliveryLoading: false,
    publishing: false,
    error: null,
  }),
  actions: {
    applyGithubAccountsPayload(payload: any) {
      this.githubAccounts = payload?.accounts || [];
      this.githubActiveAccountId = payload?.active_id || '';
      this.githubActiveAccount =
        payload?.active_account ||
        this.githubAccounts.find((account: any) => account.id === this.githubActiveAccountId) ||
        null;
      this.githubConnected = Boolean(this.githubActiveAccount?.has_token);
      this.githubHasToken = Boolean(this.githubActiveAccount?.has_token);
      this.githubUsername = this.githubActiveAccount?.username || '';
      this.githubProfile = payload?.profile || this.githubProfile;
    },
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
    async generateInterjectionsPreview(defaultPrompt: string, projectName?: string) {
      this.saving = true;
      try {
        const data = await previewBrandInterjections(defaultPrompt, projectName);
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
        this.applyGithubAccountsPayload(data);
        this.error = null;
        return data;
      } catch (err: any) {
        this.error = err?.message || 'Failed to load GitHub account';
        throw err;
      } finally {
        this.githubAccountLoading = false;
      }
    },
    async saveGithubAccount(payload: { username?: string; token: string; account_id?: string; label?: string }) {
      this.githubAccountLoading = true;
      try {
        const data = await saveBrandGithubAccount(payload);
        this.applyGithubAccountsPayload(data);
        this.error = null;
        return data;
      } finally {
        this.githubAccountLoading = false;
      }
    },
    async setGithubActiveAccount(accountId: string) {
      this.githubAccountLoading = true;
      try {
        const data = await setBrandGithubActiveAccount(accountId);
        this.applyGithubAccountsPayload(data);
        this.error = null;
        return data;
      } finally {
        this.githubAccountLoading = false;
      }
    },
    async fetchGithubRepos(username?: string) {
      this.githubRepoLoading = true;
      try {
        const data = await fetchBrandGithubRepos(username, this.githubActiveAccountId || undefined);
        this.githubRepos = data.repos || [];
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
        const data = await fetchBrandGithubBranches(repoFullName, this.githubActiveAccountId || undefined);
        this.githubBranches = data.branches || [];
        return this.githubBranches;
      } finally {
        this.githubBranchLoading = false;
      }
    },
    async fetchGithubImportStatus(jobId: string) {
      const data = await fetchBrandGithubImportStatus(jobId);
      return data;
    },
    async startDeliveryRun(payload: { project_id: string; prompt: string; mode?: string }) {
      this.deliveryLoading = true;
      try {
        const data = await startBrandDeliveryRun(payload);
        this.activeDeliveryRun = data.run || null;
        return data.run;
      } finally {
        this.deliveryLoading = false;
      }
    },
    async fetchDeliveryRuns(projectId?: string) {
      const data = await fetchBrandDeliveryRuns(projectId);
      this.deliveryRuns = data.runs || [];
      return this.deliveryRuns;
    },
    async fetchDeliveryRun(runId: string) {
      const data = await fetchBrandDeliveryRun(runId);
      this.activeDeliveryRun = data.run || null;
      return this.activeDeliveryRun;
    },
    async fetchDeliveryBacklog(runId: string) {
      const data = await fetchBrandDeliveryBacklog(runId);
      this.deliveryBacklog = data.backlog || [];
      return this.deliveryBacklog;
    },
    async fetchDeliveryGates(runId: string) {
      const data = await fetchBrandDeliveryGates(runId);
      this.deliveryGates = data.gates || [];
      return this.deliveryGates;
    },
    async fetchDeliverySessions(runId: string) {
      const data = await fetchBrandDeliverySessions(runId);
      this.deliverySessions = data.sessions || [];
      return this.deliverySessions;
    },
    async fetchDeliverySessionLogs(sessionId: string, limit = 200) {
      const data = await fetchBrandDeliverySessionLogs(sessionId, limit);
      const nextLines = data.lines || [];
      const prevLines = this.deliverySessionLogs[sessionId] || [];
      const unchanged =
        prevLines.length === nextLines.length &&
        prevLines[prevLines.length - 1] === nextLines[nextLines.length - 1];
      if (unchanged) {
        return prevLines;
      }
      this.deliverySessionLogs = {
        ...this.deliverySessionLogs,
        [sessionId]: nextLines,
      };
      return nextLines;
    },
    async fetchDeliveryArtifacts(runId: string) {
      const data = await fetchBrandDeliveryArtifacts(runId);
      this.deliveryArtifacts = data.artifacts || [];
      return this.deliveryArtifacts;
    },
    async triggerDeliveryUiCapture(runId: string) {
      this.uiCaptureLoading = true;
      try {
        const data = await triggerBrandDeliveryUiCapture(runId);
        return data;
      } finally {
        this.uiCaptureLoading = false;
      }
    },
    async fetchDeliveryGovernance(runId: string) {
      const data = await fetchBrandDeliveryGovernance(runId);
      this.deliveryGovernance = data.governance || [];
      return this.deliveryGovernance;
    },
    async fetchDeliverySprints(runId: string) {
      const data = await fetchBrandDeliverySprints(runId);
      this.deliverySprints = data.sprints || [];
      return this.deliverySprints;
    },
    async acceptDeliveryRun(runId: string, payload: { notes?: string; checklist?: string[] }) {
      const data = await acceptBrandDeliveryRun(runId, payload);
      this.activeDeliveryRun = data.run || null;
      return this.activeDeliveryRun;
    },
    async importFromGitHub(payload: Record<string, any>) {
      this.importing = true;
      try {
        const data = await importBrandProjectFromGitHub(payload);
        if (data?.project) {
          await this.load();
        }
        return data;
      } finally {
        this.importing = false;
      }
    },
    async publishProject(projectId: string, payload: Record<string, any>) {
      this.publishing = true;
      try {
        const data = await publishBrandProject(projectId, payload);
        if (data?.repo_url) {
          await this.load();
        }
        return data;
      } finally {
        this.publishing = false;
      }
    },
    async fetchGithubPublishStatus(jobId: string) {
      const data = await fetchBrandGithubPublishStatus(jobId);
      if (data?.result?.repo_url) {
        await this.load();
      }
      return data;
    },
  },
});
