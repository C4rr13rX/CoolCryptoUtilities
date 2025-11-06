import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 8000,
  withCredentials: true,
});

api.defaults.xsrfCookieName = 'csrftoken';
api.defaults.xsrfHeaderName = 'X-CSRFToken';

function getCsrfToken(): string | null {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(/csrftoken=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
}

api.interceptors.request.use((config) => {
  if (!config.headers) {
    config.headers = {};
  }
  const headerKey = api.defaults.xsrfHeaderName || 'X-CSRFToken';
  if (!config.headers[headerKey]) {
    const token = getCsrfToken();
    if (token) {
      config.headers[headerKey] = token;
    }
  }
  return config;
});

export async function fetchDashboardSummary() {
  const { data } = await api.get('/telemetry/dashboard/');
  return data;
}

export async function fetchLatestStreams(symbols?: string[]) {
  const params = new URLSearchParams();
  symbols?.forEach((symbol) => params.append('symbol', symbol));
  const { data } = await api.get('/streams/latest/', { params });
  return data;
}

export async function fetchRecentTrades(limit = 100) {
  const { data } = await api.get('/telemetry/trades/', { params: { limit } });
  return data;
}

export async function fetchFeedback(limit = 100) {
  const { data } = await api.get('/telemetry/feedback/', { params: { limit } });
  return data;
}

export async function fetchAdvisories(limit = 100, includeResolved = false) {
  const params: Record<string, string> = { limit: String(limit) };
  if (includeResolved) {
    params.include_resolved = '1';
  }
  const { data } = await api.get('/telemetry/advisories/', { params });
  return data;
}

export async function fetchMetrics(stage?: string, limit = 200) {
  const params: Record<string, string> = { limit: String(limit) };
  if (stage) params.stage = stage;
  const { data } = await api.get('/telemetry/metrics/', { params });
  return data;
}

export async function fetchOrganismLatest() {
  const { data } = await api.get('/telemetry/organism/latest/');
  return data;
}

export interface OrganismHistoryParams {
  start_ts?: number;
  end_ts?: number;
  limit?: number;
}

export async function fetchOrganismHistory(params?: OrganismHistoryParams) {
  const query: Record<string, string> = {};
  if (params?.limit) query.limit = String(params.limit);
  if (params?.start_ts) query.start_ts = String(params.start_ts);
  if (params?.end_ts) query.end_ts = String(params.end_ts);
  const { data } = await api.get('/telemetry/organism/history/', { params: query });
  return data;
}

export async function fetchConsoleStatus() {
  const { data } = await api.get('/console/status/');
  return data;
}

export async function fetchConsoleLogs(limit = 200) {
  const { data } = await api.get('/console/logs/', { params: { limit } });
  return data;
}

export async function startConsoleProcess() {
  const { data } = await api.post('/console/start/');
  return data;
}

export async function stopConsoleProcess() {
  const { data } = await api.post('/console/stop/');
  return data;
}

export async function sendConsoleInput(command: string) {
  const { data } = await api.post('/console/input/', { command });
  return data;
}
