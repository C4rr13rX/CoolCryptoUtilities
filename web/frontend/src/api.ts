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

export async function fetchOrganismConfig() {
  const { data } = await api.get('/telemetry/organism/settings/');
  return data;
}

export async function updateOrganismConfig(payload: { label_scale: number }) {
  const { data } = await api.post('/telemetry/organism/settings/', payload);
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

export interface LabJobPayload {
  train_files?: string[];
  eval_files?: string[];
  epochs?: number;
  batch_size?: number;
}

export async function fetchLabFiles() {
  const { data } = await api.get('/lab/files/');
  return data;
}

export async function fetchLabStatus() {
  const { data } = await api.get('/lab/status/');
  return data;
}

export async function startLabJob(payload: LabJobPayload) {
  const { data } = await api.post('/lab/run/', payload);
  return data;
}

export async function fetchLabNews(payload: { train_files?: string[]; eval_files?: string[] }) {
  const { data } = await api.post('/lab/news/', payload);
  return data;
}

export interface LabPreviewPayload {
  files: string[];
  batch_size?: number;
  include_news?: boolean;
}

export async function fetchLabPreview(payload: LabPreviewPayload) {
  const { data } = await api.post('/lab/preview/', payload, { timeout: 60000 });
  return data;
}

export interface DataLabDatasetParams {
  chain?: string;
  category?: string;
  sort?: string;
  order?: string;
}

export async function fetchDataLabDatasets(params?: DataLabDatasetParams) {
  const { data } = await api.get('/datalab/datasets/', { params });
  return data;
}

export interface DataLabJobPayload {
  job_type: string;
  options?: Record<string, any>;
}

export async function startDataLabJob(payload: DataLabJobPayload) {
  const { data } = await api.post('/datalab/run/', payload);
  return data;
}

export async function fetchDataLabJobStatus() {
  const { data } = await api.get('/datalab/status/');
  return data;
}

export interface DataLabNewsPayload {
  tokens: string[];
  start: string;
  end: string;
  query?: string;
  max_pages?: number;
}

export async function fetchDataLabNews(payload: DataLabNewsPayload) {
  const { data } = await api.post('/datalab/news/', payload);
  return data;
}

export interface GuardianSettingsPayload {
  enabled?: boolean;
  default_prompt?: string;
  interval_minutes?: number;
}

export async function fetchGuardianSettings() {
  const { data } = await api.get('/guardian/settings/');
  return data;
}

export async function updateGuardianSettings(payload: GuardianSettingsPayload) {
  const { data } = await api.post('/guardian/settings/', payload);
  return data;
}

export async function runGuardianJob(payload: { prompt?: string; save_default?: boolean }) {
  const { data } = await api.post('/guardian/run/', payload);
  return data;
}

export async function fetchGuardianLogs(limit = 200) {
  const { data } = await api.get('/guardian/logs/', { params: { limit } });
  return data;
}

export interface SecureSettingPayload {
  id?: number;
  name: string;
  category?: string;
  is_secret: boolean;
  value?: string;
}

export async function fetchSecureSettings(params?: { reveal?: boolean }) {
  const { data } = await api.get('/secure/settings/', { params: params?.reveal ? { reveal: '1' } : undefined });
  return data;
}

export async function createSecureSetting(payload: SecureSettingPayload) {
  const { data } = await api.post('/secure/settings/', payload);
  return data;
}

export async function updateSecureSetting(id: number, payload: SecureSettingPayload) {
  const { data } = await api.patch(`/secure/settings/${id}/`, payload);
  return data;
}

export async function deleteSecureSetting(id: number) {
  const { data } = await api.delete(`/secure/settings/${id}/`);
  return data;
}

export async function revealSecureSetting(id: number) {
  const { data } = await api.get(`/secure/settings/${id}/`, { params: { reveal: '1' } });
  return data;
}

export interface SecureSettingImportPayload {
  content: string;
  category?: string;
  is_secret?: boolean;
}

export async function importSecureSettings(payload: SecureSettingImportPayload) {
  const { data } = await api.post('/secure/settings/import/', payload);
  return data;
}

export async function clearSecureSettings(category?: string) {
  const { data } = await api.delete('/secure/settings/', { params: category ? { category } : undefined });
  return data;
}

export async function fetchWalletStatus() {
  const { data } = await api.get('/wallet/actions/');
  return data;
}

export async function runWalletAction(payload: { action: string; options?: Record<string, any> }) {
  const { data } = await api.post('/wallet/run/', payload);
  return data;
}

export async function fetchWalletMnemonic() {
  const { data } = await api.get('/wallet/mnemonic/');
  return data;
}

export async function updateWalletMnemonic(mnemonic: string | null) {
  const { data } = await api.post('/wallet/mnemonic/', { mnemonic });
  return data;
}

export async function fetchWalletState() {
  const { data } = await api.get('/wallet/state/');
  return data;
}

export async function fetchIntegrations() {
  const { data } = await api.get('/integrations/keys/');
  return data;
}

export async function updateIntegration(name: string, value: string | null) {
  const { data } = await api.post(`/integrations/keys/${name}/`, { value });
  return data;
}

export async function testIntegration(name: string, value: string) {
  const { data } = await api.post(`/integrations/keys/${name}/test/`, { value });
  return data;
}

export async function revealIntegrationValue(name: string) {
  const { data } = await api.get(`/integrations/keys/${name}/`, { params: { reveal: '1' } });
  return data;
}

export async function fetchCodeGraph(refresh = false) {
  const params = refresh ? { refresh: '1' } : undefined;
  const { data } = await api.get('/codegraph/', { params, timeout: 30000 });
  return data;
}

export async function fetchCodeGraphFiles() {
  const { data } = await api.get('/codegraph/files/', { timeout: 10000 });
  return data?.files || [];
}

export async function uploadCodeGraphSnapshot(payload: { timestamp: string; node_id: string; image: string }) {
  const { data } = await api.post('/codegraph/snapshots/', payload);
  return data;
}

// ----------------------------- BrandDozer ------------------------------------
export async function fetchBrandProjects() {
  const { data } = await api.get('/branddozer/projects/');
  return data;
}

export async function fetchBrandRoots(path?: string) {
  const params = path ? { path } : undefined;
  const { data } = await api.get('/branddozer/projects/roots/', { params });
  return data;
}

export async function fetchBrandGithubAccount() {
  const { data } = await api.get('/branddozer/projects/github/accounts/');
  return data;
}

export async function saveBrandGithubAccount(payload: { username?: string; token: string; account_id?: string; label?: string }) {
  const { data } = await api.post('/branddozer/projects/github/accounts/', payload);
  return data;
}

export async function setBrandGithubActiveAccount(accountId: string) {
  const { data } = await api.post('/branddozer/projects/github/accounts/active/', { account_id: accountId });
  return data;
}

export async function fetchBrandGithubRepos(username?: string, accountId?: string) {
  const params: Record<string, string> = {};
  if (username) params.username = username;
  if (accountId) params.account_id = accountId;
  const { data } = await api.get('/branddozer/projects/github/repos/', { params, timeout: 12000 });
  return data;
}

export async function fetchBrandGithubBranches(repoFullName: string, accountId?: string) {
  const params: Record<string, string> = { repo: repoFullName };
  if (accountId) params.account_id = accountId;
  const { data } = await api.get('/branddozer/projects/github/branches/', { params });
  return data;
}

export async function createBrandProject(payload: Record<string, any>) {
  const { data } = await api.post('/branddozer/projects/', payload);
  return data;
}

export async function updateBrandProject(id: string, payload: Record<string, any>) {
  const { data } = await api.patch(`/branddozer/projects/${id}/`, payload);
  return data;
}

export async function deleteBrandProject(id: string) {
  const { data } = await api.delete(`/branddozer/projects/${id}/`);
  return data;
}

export async function startBrandProject(id: string) {
  const { data } = await api.post(`/branddozer/projects/${id}/start/`);
  return data;
}

export async function stopBrandProject(id: string) {
  const { data } = await api.post(`/branddozer/projects/${id}/stop/`);
  return data;
}

export async function fetchBrandLogs(id: string, limit = 200) {
  const { data } = await api.get(`/branddozer/projects/${id}/logs/`, { params: { limit } });
  return data;
}

export async function generateBrandInterjections(id: string, defaultPrompt?: string) {
  const payload = defaultPrompt ? { default_prompt: defaultPrompt } : undefined;
  const { data } = await api.post(`/branddozer/projects/${id}/interjections/`, payload);
  return data;
}

export async function previewBrandInterjections(defaultPrompt: string, projectName?: string) {
  const payload = { default_prompt: defaultPrompt, project_name: projectName };
  const { data } = await api.post('/branddozer/projects/interjections/preview/', payload);
  return data;
}

export async function importBrandProjectFromGitHub(payload: Record<string, any>) {
  const { data } = await api.post('/branddozer/projects/import/github/', payload, { timeout: 120000 });
  return data;
}

export async function fetchBrandGithubImportStatus(jobId: string) {
  const { data } = await api.get(`/branddozer/projects/import/github/status/${jobId}/`, { timeout: 120000 });
  return data;
}

export async function publishBrandProject(projectId: string, payload: Record<string, any>) {
  const { data } = await api.post(`/branddozer/projects/${projectId}/publish/`, payload, { timeout: 120000 });
  return data;
}

export async function startBrandDeliveryRun(payload: { project_id: string; prompt: string; mode?: string }) {
  const { data } = await api.post('/branddozer/delivery/runs/', payload, { timeout: 120000 });
  return data;
}

export async function fetchBrandDeliveryRuns(projectId?: string) {
  const params = projectId ? { project_id: projectId } : undefined;
  const { data } = await api.get('/branddozer/delivery/runs/', { params });
  return data;
}

export async function fetchBrandDeliveryRun(runId: string) {
  const { data } = await api.get(`/branddozer/delivery/runs/${runId}/`);
  return data;
}

export async function fetchBrandDeliveryBacklog(runId: string) {
  const { data } = await api.get(`/branddozer/delivery/runs/${runId}/backlog/`);
  return data;
}

export async function updateBrandDeliveryBacklogItem(itemId: string, payload: { status?: string; priority?: number }) {
  const { data } = await api.patch(`/branddozer/delivery/backlog/${itemId}/`, payload);
  return data;
}

export async function fetchBrandDeliveryGates(runId: string) {
  const { data } = await api.get(`/branddozer/delivery/runs/${runId}/gates/`);
  return data;
}

export async function fetchBrandDeliverySessions(runId: string) {
  const { data } = await api.get(`/branddozer/delivery/runs/${runId}/sessions/`);
  return data;
}

export async function fetchBrandDeliverySessionLogs(sessionId: string, limit = 200) {
  const { data } = await api.get(`/branddozer/delivery/sessions/${sessionId}/logs/`, { params: { limit } });
  return data;
}

export async function fetchBrandDeliveryArtifacts(runId: string) {
  const { data } = await api.get(`/branddozer/delivery/runs/${runId}/artifacts/`);
  return data;
}

export async function triggerBrandDeliveryUiCapture(runId: string) {
  const { data } = await api.post(`/branddozer/delivery/runs/${runId}/ui-capture/`);
  return data;
}

export async function fetchBrandDeliveryGovernance(runId: string) {
  const { data } = await api.get(`/branddozer/delivery/runs/${runId}/governance/`);
  return data;
}

export async function fetchBrandDeliverySprints(runId: string) {
  const { data } = await api.get(`/branddozer/delivery/runs/${runId}/sprints/`);
  return data;
}

export async function acceptBrandDeliveryRun(runId: string, payload: { notes?: string; checklist?: string[] }) {
  const { data } = await api.post(`/branddozer/delivery/runs/${runId}/accept/`, payload);
  return data;
}

// ----------------------------- U53RxR080T (UX robot) -------------------------
export async function fetchUxAgents() {
  const { data } = await api.get('/u53rxr080t/agents/');
  return data;
}

export async function fetchUxTasks(status?: string) {
  const params = status ? { status } : undefined;
  const { data } = await api.get('/u53rxr080t/tasks/', { params });
  return data;
}

export async function fetchUxFindings(limit = 50) {
  const { data } = await api.get('/u53rxr080t/findings/', { params: { limit } });
  return data;
}

export async function createUxTask(payload: Record<string, any>) {
  const { data } = await api.post('/u53rxr080t/tasks/', payload);
  return data;
}

export async function updateUxTask(id: string, payload: Record<string, any>) {
  const { data } = await api.post(`/u53rxr080t/tasks/${id}/`, payload);
  return data;
}

export async function claimNextUxTask(agentId: string) {
  const { data } = await api.post('/u53rxr080t/tasks/next/', { agent_id: agentId });
  return data;
}

export async function sendUxFinding(payload: Record<string, any>) {
  const { data } = await api.post('/u53rxr080t/findings/', payload);
  return data;
}

export interface DataLabSignalParams {
  window?: string;
  direction?: 'bullish' | 'bearish' | 'all';
  limit?: number;
  min_volume?: number;
}

export async function fetchDataLabSignals(params?: DataLabSignalParams) {
  const { data } = await api.get('/datalab/signals/', { params });
  return data;
}

export interface DataLabWatchlistUpdatePayload {
  target: 'stream' | 'ghost' | 'live';
  action?: 'add' | 'remove' | 'set' | 'replace';
  symbols: string[];
}

export async function fetchDataLabWatchlists() {
  const { data } = await api.get('/datalab/watchlists/');
  return data;
}

export async function updateDataLabWatchlist(payload: DataLabWatchlistUpdatePayload) {
  const { data } = await api.post('/datalab/watchlists/', payload);
  return data;
}
