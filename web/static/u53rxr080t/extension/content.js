// Content script: executes click steps on demand.
function clickSelector(selector) {
  const el = document.querySelector(selector);
  if (!el) return { ok: false, detail: 'not found' };
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  el.click();
  return { ok: true };
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === 'u53rx-click') {
    try {
      const res = clickSelector(msg.selector);
      sendResponse(res);
    } catch (err) {
      sendResponse({ ok: false, detail: String(err) });
    }
    return true;
  }
  return undefined;
});
