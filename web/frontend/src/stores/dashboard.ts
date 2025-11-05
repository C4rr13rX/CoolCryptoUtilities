import { defineStore } from 'pinia';
import {
  fetchConsoleLogs,
  fetchConsoleStatus,
  fetchDashboardSummary,
  fetchLatestStreams,
  fetchAdvisories,
  startConsoleProcess,
  stopConsoleProcess
} from '@/api';

interface DashboardState {
  dashboard: Record<string, any> | null;
  streams: Record<string, any>;
  consoleStatus: Record<string, any> | null;
  consoleLogs: string[];
  advisories: any[];
  loading: boolean;
  error: string | null;
}

export const useDashboardStore = defineStore('dashboard', {
  state: (): DashboardState => ({
    dashboard: null,
    streams: {},
    consoleStatus: null,
    consoleLogs: [],
    advisories: [],
    loading: false,
    error: null
  }),
  actions: {
    async refreshAll() {
      this.loading = true;
      try {
        const [summary, streams, consoleStatus, consoleLogs, advisories] = await Promise.all([
          fetchDashboardSummary(),
          fetchLatestStreams(),
          fetchConsoleStatus(),
          fetchConsoleLogs(200),
          fetchAdvisories(50)
        ]);
        this.dashboard = summary;
        this.streams = streams;
        this.consoleStatus = consoleStatus;
        this.consoleLogs = consoleLogs.lines || [];
        this.advisories = advisories;
        this.error = null;
      } catch (error: any) {
        this.error = error?.message || 'Failed to refresh data';
      } finally {
        this.loading = false;
      }
    },
    async refreshConsole() {
      const [consoleStatus, consoleLogs] = await Promise.all([
        fetchConsoleStatus(),
        fetchConsoleLogs(200)
      ]);
      this.consoleStatus = consoleStatus;
      this.consoleLogs = consoleLogs.lines || [];
    },
    async startProcess() {
      await startConsoleProcess();
      await this.refreshConsole();
    },
    async stopProcess() {
      await stopConsoleProcess();
      await this.refreshConsole();
    },
  }
});
