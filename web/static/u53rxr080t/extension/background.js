// U53RxR080T background: polls dashboard API for tasks, runs scripted clicks, captures screenshots, sends findings.
const DEFAULT_CONFIG = {
  serverUrl: 'http://127.0.0.1:8000',
  agentName: 'browser-extension',
  authToken: '', // optional bearer token; cookies will also be sent
  enabled: false, // opt-in so it isn't always running
  daemonUrl: 'http://127.0.0.1:36279',
};

let cachedConfig = { ...DEFAULT_CONFIG };
let agentId = null;
let polling = null;

function log(...args) {
  console.log('[u53rx]', ...args);
}

async function loadConfig() {
  const stored = await chrome.storage.sync.get(Object.keys(DEFAULT_CONFIG));
  cachedConfig = { ...DEFAULT_CONFIG, ...stored };
}

async function saveConfig(cfg) {
  await chrome.storage.sync.set(cfg);
  cachedConfig = { ...cachedConfig, ...cfg };
}

async function callDaemon(path, body) {
  try {
    const url = `${cachedConfig.daemonUrl.replace(/\\/$/, '')}${path}`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!resp.ok) throw new Error(`Daemon ${resp.status}`);
    return await resp.json();
  } catch (err) {
    log('daemon call failed', err);
    return null;
  }
}

async function ensureAgentId() {
  if (agentId) return agentId;
  const stored = await chrome.storage.sync.get(['agentId']);
  if (stored.agentId) {
    agentId = stored.agentId;
    return agentId;
  }
  // lightweight UUID
  agentId = crypto.randomUUID();
  await chrome.storage.sync.set({ agentId });
  return agentId;
}

async function apiFetch(path, body) {
  const url = `${cachedConfig.serverUrl.replace(/\/$/, '')}${path}`;
  const headers = { 'Content-Type': 'application/json' };
  if (cachedConfig.authToken) headers['Authorization'] = `Bearer ${cachedConfig.authToken}`;
  const resp = await fetch(url, {
    method: 'POST',
    headers,
    credentials: 'include',
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API ${resp.status}: ${text}`);
  }
  return resp.json();
}

async function heartbeat(status = 'idle', meta = {}) {
  const id = await ensureAgentId();
  try {
    await apiFetch('/api/u53rxr080t/heartbeat/', {
      id,
      name: cachedConfig.agentName,
      kind: 'browser',
      platform: navigator.platform,
      browser: navigator.userAgent,
      status,
      meta,
    });
  } catch (err) {
    log('heartbeat failed', err);
  }
}

async function claimTask() {
  const id = await ensureAgentId();
  return apiFetch('/api/u53rxr080t/tasks/next/', { agent_id: id });
}

async function updateTask(taskId, status, meta = {}) {
  return apiFetch(`/api/u53rxr080t/tasks/${taskId}/`, { status, meta, assigned_to: agentId });
}

async function sendFinding(task, screenshotDataUrl, context = {}) {
  const id = await ensureAgentId();
  try {
    await apiFetch('/api/u53rxr080t/findings/', {
      session: id,
      title: task?.title || 'UX finding',
      summary: context.note || 'Captured screenshot',
      severity: context.severity || 'info',
      screenshot_url: screenshotDataUrl,
      context,
    });
  } catch (err) {
    log('finding failed', err);
  }
}

async function enqueueGuardian(message, meta = {}) {
  const id = await ensureAgentId();
  try {
    await apiFetch('/api/u53rxr080t/queue/', {
      session: id,
      title: meta.title || message || 'UX robot update',
      summary: message || '',
      severity: meta.severity || 'info',
      meta,
    });
  } catch (err) {
    log('guardian queue failed', err);
  }
}

async function requestSuggestions(task, screenshotDataUrl, context = {}) {
  try {
    return await apiFetch('/api/u53rxr080t/suggest/', {
      task,
      screenshot: screenshotDataUrl,
      context,
    });
  } catch (err) {
    log('suggest failed', err);
    return null;
  }
}

function activeTab() {
  return chrome.tabs.query({ active: true, currentWindow: true }).then((tabs) => tabs[0]);
}

async function capture(tabId) {
  const dataUrl = await chrome.tabs.captureVisibleTab(undefined, { format: 'png' });
  return dataUrl;
}

async function daemonSequence() {
  const resp = await callDaemon('/sequence', { count: 3, interval_ms: 300 });
  if (resp?.frames?.length) {
    return resp.frames.map((f) => `data:image/${f.format || 'png'};base64,${f.image}`);
  }
  return null;
}

async function runClicks(tabId, actions = []) {
  if (!actions.length) return { ok: true, steps: [] };
  const results = [];
  for (const step of actions) {
    try {
      const res = await chrome.tabs.sendMessage(tabId, { type: 'u53rx-click', selector: step.selector || step });
      results.push({ selector: step.selector || step, ok: !!res?.ok, detail: res?.detail || '' });
    } catch (err) {
      results.push({ selector: step.selector || step, ok: false, detail: String(err) });
    }
    await new Promise((r) => setTimeout(r, step.delay_ms || 700));
  }
  return { ok: results.every((r) => r.ok), steps: results };
}

async function processTask(task) {
  const tab = await activeTab();
  if (!tab) {
    log('no active tab to run task');
    return;
  }
  await updateTask(task.id, 'in_progress', { started_at: Date.now() });
  const actions = (task.meta && task.meta.actions) || [];
  const run = await runClicks(tab.id, actions);
  let screenshots = await daemonSequence();
  if (!screenshots || !screenshots.length) {
    screenshots = [];
    for (let i = 0; i < Math.max(1, actions.length); i++) {
      screenshots.push(await capture(tab.id));
      await new Promise((r) => setTimeout(r, 250));
    }
  }
  const latestShot = screenshots[screenshots.length - 1];
  const suggestion = await requestSuggestions(task, latestShot, { actions: run.steps, stage: task.stage });
  await sendFinding(task, latestShot, {
    note: 'Auto-run UX task',
    actions: run.steps,
    stage: task.stage,
    suggestions: suggestion?.actions || [],
  });
  await updateTask(task.id, run.ok ? 'done' : 'error', {
    steps: run.steps,
    screenshots: screenshots.length,
    suggestions: suggestion?.actions || [],
  });
  if (suggestion?.actions?.length) {
    await enqueueGuardian(`Suggested next actions for ${task.title}`, { actions: suggestion.actions, severity: 'info' });
  }
}

async function pollLoop() {
  if (polling) clearInterval(polling);
  polling = setInterval(async () => {
    if (!cachedConfig.enabled) return;
    try {
      await heartbeat('idle');
      const { task } = await claimTask();
      if (task) {
        log('claimed task', task.id);
        await processTask(task);
        await heartbeat('idle');
      }
    } catch (err) {
      log('poll error', err);
    }
  }, 15000);
}

chrome.runtime.onInstalled.addListener(async () => {
  await loadConfig();
  await ensureAgentId();
  await heartbeat('idle', { event: 'installed' });
  await pollLoop();
});

chrome.runtime.onStartup.addListener(async () => {
  await loadConfig();
  await ensureAgentId();
  await heartbeat('idle', { event: 'startup' });
  await pollLoop();
});

chrome.storage.onChanged.addListener(() => {
  loadConfig().then(pollLoop);
});

// Respond to manual trigger from popup (future) if needed.
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === 'u53rx-refresh-config') {
    loadConfig().then(() => sendResponse({ ok: true }));
    return true;
  }
  return undefined;
});
