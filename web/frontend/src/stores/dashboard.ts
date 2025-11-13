import { defineStore } from 'pinia';
import {
  fetchConsoleLogs,
  fetchConsoleStatus,
  fetchDashboardSummary,
  fetchLatestStreams,
  fetchAdvisories,
  fetchRecentTrades,
  fetchFeedback,
  fetchMetrics,
  startConsoleProcess,
  stopConsoleProcess,
  sendConsoleInput,
  fetchGuardianLogs,
} from '@/api';

interface DashboardState {
  dashboard: Record<string, any> | null;
  streams: Record<string, any>;
  consoleStatus: Record<string, any> | null;
  consoleLogs: string[];
  guardianLogs: string[];
  advisories: any[];
  recentTrades: any[];
  latestFeedback: any[];
  latestMetrics: any[];
  loading: boolean;
  error: string | null;
}

export const useDashboardStore = defineStore('dashboard', {
  state: (): DashboardState => ({
    dashboard: null,
    streams: {},
    consoleStatus: null,
    consoleLogs: [],
    guardianLogs: [],
    advisories: [],
    recentTrades: [],
    latestFeedback: [],
    latestMetrics: [],
    loading: false,
    error: null,
  }),
  actions: {
    hydrateFromSnapshot(snapshot: Record<string, any>) {
      if (!snapshot) return;
      const safe = snapshot || {};
      this.dashboard = { ...safe };
      this.latestMetrics = safe.latest_metrics || [];
      this.latestFeedback = safe.latest_feedback || [];
      const ghost = safe.ghost_trades || [];
      const live = safe.live_trades || [];
      this.recentTrades = [...ghost, ...live];
      this.advisories = safe.advisories || safe.active_advisories || [];
    },
    async refreshAll() {
      if (this.loading) {
        return;
      }
      this.loading = true;
      try {
        const [
          summary,
          streams,
          consoleStatus,
          consoleLogs,
          guardianLogs,
          advisories,
          trades,
          feedback,
          metrics,
        ] = await Promise.all([
          fetchDashboardSummary(),
          fetchLatestStreams(),
          fetchConsoleStatus(),
          fetchConsoleLogs(200),
          fetchGuardianLogs(200),
          fetchAdvisories(50),
          fetchRecentTrades(50),
          fetchFeedback(50),
          fetchMetrics(undefined, 50),
        ]);
        this.dashboard = summary || {};
        this.streams = streams;
        this.consoleStatus = consoleStatus;
        this.consoleLogs = consoleLogs.lines || [];
        this.guardianLogs = guardianLogs.lines || [];
        const advisoryList = advisories?.results || advisories || summary?.active_advisories || [];
        this.advisories = advisoryList;
        this.recentTrades = trades?.results || trades || summary?.recent_trades || [];
        this.latestFeedback = feedback?.results || feedback || summary?.latest_feedback || [];
        this.latestMetrics = metrics?.results || metrics || summary?.latest_metrics || [];
        this.error = null;
      } catch (error: any) {
        this.error = error?.message || 'Failed to refresh data';
      } finally {
        this.loading = false;
      }
    },
    async refreshConsole() {
      try {
        const [consoleStatus, consoleLogs, guardianLogs] = await Promise.all([
          fetchConsoleStatus(),
          fetchConsoleLogs(200),
          fetchGuardianLogs(200),
        ]);
        this.consoleStatus = consoleStatus;
        this.consoleLogs = consoleLogs.lines || [];
        this.guardianLogs = guardianLogs.lines || [];
        this.error = null;
      } catch (err: any) {
        this.error = err?.message || 'Failed to refresh console';
      }
    },
    async startProcess() {
      await startConsoleProcess();
      await this.refreshConsole();
    },
    async stopProcess() {
      await stopConsoleProcess();
      await this.refreshConsole();
    },
    async sendConsoleInput(command: string) {
      if (!command) return;
      await sendConsoleInput(command);
      await this.refreshConsole();
    },
  }
});
