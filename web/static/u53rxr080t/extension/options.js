const serverUrlEl = document.getElementById('serverUrl');
const daemonUrlEl = document.getElementById('daemonUrl');
const agentNameEl = document.getElementById('agentName');
const authTokenEl = document.getElementById('authToken');
const statusEl = document.getElementById('status');
const enabledEl = document.getElementById('enabled');

const DEFAULTS = {
  serverUrl: 'http://127.0.0.1:8000',
  daemonUrl: 'http://127.0.0.1:36279',
  agentName: 'browser-extension',
  authToken: '',
  enabled: false,
};

function load() {
  chrome.storage.sync.get(Object.keys(DEFAULTS), (items) => {
    serverUrlEl.value = items.serverUrl || DEFAULTS.serverUrl;
    daemonUrlEl.value = items.daemonUrl || DEFAULTS.daemonUrl;
    agentNameEl.value = items.agentName || DEFAULTS.agentName;
    authTokenEl.value = items.authToken || '';
    enabledEl.checked = Boolean(items.enabled);
  });
}

function save() {
  const cfg = {
    serverUrl: serverUrlEl.value.trim() || DEFAULTS.serverUrl,
    daemonUrl: daemonUrlEl.value.trim() || DEFAULTS.daemonUrl,
    agentName: agentNameEl.value.trim() || DEFAULTS.agentName,
    authToken: authTokenEl.value.trim(),
    enabled: enabledEl.checked,
  };
  chrome.storage.sync.set(cfg, () => {
    statusEl.textContent = 'Saved';
    chrome.runtime.sendMessage({ type: 'u53rx-refresh-config' });
    setTimeout(() => (statusEl.textContent = ''), 1200);
  });
}

document.getElementById('save').addEventListener('click', save);

document.addEventListener('DOMContentLoaded', load);
