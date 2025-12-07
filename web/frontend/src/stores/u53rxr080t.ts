import { defineStore } from 'pinia';
import {
  fetchUxAgents,
  fetchUxTasks,
  fetchUxFindings,
  createUxTask,
  updateUxTask,
  claimNextUxTask,
} from '@/api';

export interface UxAgent {
  id: string;
  name: string;
  kind: string;
  platform?: string;
  browser?: string;
  status: string;
  last_seen?: string;
  meta?: Record<string, any>;
}

export interface UxTask {
  id: string;
  title: string;
  description: string;
  stage: string;
  status: string;
  target_url?: string;
  assigned_to?: string | null;
  meta?: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}

export interface UxFinding {
  id: string;
  session: string;
  title: string;
  summary: string;
  severity: string;
  screenshot_url?: string;
  context?: Record<string, any>;
  created_at?: string;
}

export const useUxRobotStore = defineStore('u53rxr080t', {
  state: () => ({
    loading: false,
    error: '' as string,
    agents: [] as UxAgent[],
    tasks: [] as UxTask[],
    findings: [] as UxFinding[],
    lastUpdated: 0,
  }),
  actions: {
    setError(msg: string) {
      this.error = msg;
    },
    async refreshAll() {
      this.loading = true;
      try {
        const [agents, tasks, findings] = await Promise.all([
          fetchUxAgents(),
          fetchUxTasks(),
          fetchUxFindings(100),
        ]);
        this.agents = agents?.agents || [];
        this.tasks = tasks?.tasks || [];
        this.findings = findings?.findings || [];
        this.lastUpdated = Date.now();
        this.error = '';
      } catch (error: any) {
        this.error = error?.message || 'Failed to load UX robot data';
      } finally {
        this.loading = false;
      }
    },
    async refreshTasks() {
      try {
        const { tasks } = await fetchUxTasks();
        this.tasks = tasks || [];
      } catch (error: any) {
        this.setError(error?.message || 'Failed to refresh tasks');
      }
    },
    async addTask(payload: Partial<UxTask>) {
      await createUxTask(payload as Record<string, any>);
      await this.refreshTasks();
    },
    async updateTask(id: string, payload: Partial<UxTask>) {
      await updateUxTask(id, payload as Record<string, any>);
      await this.refreshTasks();
    },
    async claimNext(agentId: string) {
      const { task } = await claimNextUxTask(agentId);
      if (task) {
        await this.refreshTasks();
      }
      return task as UxTask | null;
    },
  },
});
