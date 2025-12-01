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
  serverOnline: boolean;
  lastConsoleAttempt: number;
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
    serverOnline: true,
    lastConsoleAttempt: 0,
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
          guardianLogsResp,
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
        const guardianLines = guardianLogsResp.guardian_lines || guardianLogsResp.lines || [];
        const guardianProduction = guardianLogsResp.production_lines || [];
        const consoleLineList = consoleLogs.lines || [];
        this.consoleLogs = guardianProduction.length ? guardianProduction : consoleLineList;
        this.guardianLogs = guardianLines;
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
    async refreshConsole(force = false) {
      const now = Date.now();
      if (!force && !this.serverOnline && now - this.lastConsoleAttempt < 15000) {
        return;
      }
      this.lastConsoleAttempt = now;
      try {
        const [consoleStatus, consoleLogs, guardianLogsResp] = await Promise.all([
          fetchConsoleStatus(),
          fetchConsoleLogs(200),
          fetchGuardianLogs(200),
        ]);
        this.consoleStatus = consoleStatus;
        const guardianLines = guardianLogsResp.guardian_lines || guardianLogsResp.lines || [];
        const guardianProduction = guardianLogsResp.production_lines || [];
        const consoleLineList = consoleLogs.lines || [];
        this.consoleLogs = guardianProduction.length ? guardianProduction : consoleLineList;
        this.guardianLogs = guardianLines;
        this.error = null;
        this.serverOnline = true;
      } catch (err: any) {
        this.serverOnline = false;
        this.consoleStatus = { status: 'offline', pid: null, uptime: null };
        this.error = err?.message || 'Console offline';
      }
    },
    async startProcess() {
      await startConsoleProcess();
      await this.refreshConsole(true);
    },
    async stopProcess() {
      await stopConsoleProcess();
      await this.refreshConsole(true);
    },
    async sendConsoleInput(command: string) {
      if (!command) return;
      await sendConsoleInput(command);
      await this.refreshConsole(true);
    },
  }
});
