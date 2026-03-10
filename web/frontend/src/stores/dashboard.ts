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
        const results = await Promise.allSettled([
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
        const val = (i: number, fallback: any = null) => results[i].status === 'fulfilled' ? (results[i] as PromiseFulfilledResult<any>).value : fallback;
        const summary = val(0, this.dashboard);
        const streams = val(1, this.streams);
        const consoleStatus = val(2, this.consoleStatus);
        const consoleLogs = val(3, { lines: [] });
        const guardianLogsResp = val(4, { guardian_lines: [], production_lines: [] });
        const advisories = val(5, null);
        const trades = val(6, null);
        const feedback = val(7, null);
        const metrics = val(8, null);
        this.dashboard = summary || {};
        this.streams = streams || {};
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
        const anyFailed = results.some(r => r.status === 'rejected');
        const allFailed = results.every(r => r.status === 'rejected');
        this.serverOnline = !allFailed;
        this.error = anyFailed && !allFailed ? 'Some endpoints unavailable' : allFailed ? 'Server offline' : null;
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
        const results = await Promise.allSettled([
          fetchConsoleStatus(),
          fetchConsoleLogs(200),
          fetchGuardianLogs(200),
        ]);
        const val = (i: number, fallback: any = null) => results[i].status === 'fulfilled' ? (results[i] as PromiseFulfilledResult<any>).value : fallback;
        const consoleStatus = val(0, this.consoleStatus);
        const consoleLogs = val(1, { lines: [] });
        const guardianLogsResp = val(2, { guardian_lines: [], production_lines: [] });
        if (consoleStatus) this.consoleStatus = consoleStatus;
        const guardianLines = guardianLogsResp.guardian_lines || guardianLogsResp.lines || [];
        const guardianProduction = guardianLogsResp.production_lines || [];
        const consoleLineList = consoleLogs.lines || [];
        this.consoleLogs = guardianProduction.length ? guardianProduction : consoleLineList;
        this.guardianLogs = guardianLines;
        const anySucceeded = results.some(r => r.status === 'fulfilled');
        if (anySucceeded) {
          this.serverOnline = true;
          this.error = null;
        } else {
          this.serverOnline = false;
          this.error = 'Console offline';
        }
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
