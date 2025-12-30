type SessionEntry = {
  id: string;
};

type WorkerConfig = {
  intervalMs: number;
  maxBytes: number;
  concurrency: number;
  runId: string;
};

type WorkerMessage =
  | { type: 'start'; sessions: SessionEntry[]; config: WorkerConfig }
  | { type: 'update'; sessions: SessionEntry[]; config: WorkerConfig }
  | { type: 'stop' }
  | { type: 'reset'; runId: string };

const state = {
  sessions: [] as SessionEntry[],
  cursorMap: {} as Record<string, number>,
  timer: null as ReturnType<typeof setTimeout> | null,
  running: false,
  config: {
    intervalMs: 6000,
    maxBytes: 200000,
    concurrency: 3,
    runId: '',
  } as WorkerConfig,
};

function resetCursors(runId = '') {
  state.cursorMap = {};
  state.config.runId = runId;
}

function setSessions(next: SessionEntry[]) {
  state.sessions = next || [];
  const allowed = new Set(state.sessions.map((session) => session.id));
  const nextMap: Record<string, number> = {};
  Object.entries(state.cursorMap).forEach(([key, value]) => {
    if (allowed.has(key)) {
      nextMap[key] = value;
    }
  });
  state.cursorMap = nextMap;
}

async function runPool<T>(tasks: Array<() => Promise<T>>, limit: number) {
  const results: T[] = [];
  const pending = tasks.slice();
  const runners = Array.from({ length: Math.max(1, limit) }).map(async () => {
    while (pending.length) {
      const task = pending.shift();
      if (!task) break;
      try {
        results.push(await task());
      } catch (err) {
        // Ignore per-task errors; main thread can retry next tick.
      }
    }
  });
  await Promise.all(runners);
  return results;
}

async function fetchLogs(sessionId: string) {
  let cursor = state.cursorMap[sessionId] ?? 0;
  let batches = 0;
  while (batches < 3) {
    const url = new URL(`/api/branddozer/delivery/sessions/${sessionId}/logs/`, self.location.origin);
    url.searchParams.set('cursor', String(cursor));
    url.searchParams.set('max_bytes', String(state.config.maxBytes));
    const response = await fetch(url.toString(), { credentials: 'include' });
    if (!response.ok) {
      throw new Error(`Log fetch failed (${response.status})`);
    }
    const data = await response.json();
    const lines = Array.isArray(data?.lines) ? data.lines : [];
    const nextCursor = Number.isFinite(data?.cursor) ? Number(data.cursor) : cursor;
    const hasMore = Boolean(data?.has_more);
    state.cursorMap[sessionId] = nextCursor;
    if (lines.length) {
      (self as DedicatedWorkerGlobalScope).postMessage({
        type: 'logs',
        sessionId,
        lines,
        cursor: nextCursor,
        reset: cursor === 0,
        hasMore,
      });
    }
    batches += 1;
    if (!hasMore || nextCursor === cursor) {
      break;
    }
    cursor = nextCursor;
  }
}

async function pollOnce() {
  if (!state.sessions.length) return;
  const tasks = state.sessions.map((session) => () => fetchLogs(session.id));
  await runPool(tasks, state.config.concurrency);
}

async function tick() {
  if (!state.running) return;
  await pollOnce();
  if (!state.running) return;
  state.timer = setTimeout(tick, state.config.intervalMs);
}

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  if (!message) return;
  if (message.type === 'stop') {
    state.running = false;
    if (state.timer) {
      clearTimeout(state.timer);
      state.timer = null;
    }
    return;
  }
  if (message.type === 'reset') {
    resetCursors(message.runId);
    return;
  }
  if (message.type === 'start' || message.type === 'update') {
    const { sessions, config } = message;
    if (config && config.runId && config.runId !== state.config.runId) {
      resetCursors(config.runId);
    }
    state.config = { ...state.config, ...config };
    setSessions(sessions);
    if (message.type === 'start') {
      state.running = true;
      if (state.timer) {
        clearTimeout(state.timer);
        state.timer = null;
      }
      tick();
    }
  }
};
