type AppRealtimeHooks = {
  onSnapshot: (summary: any, walletSnapshot: any, reconciliation: any, revision: string) => void;
  onFallbackHeartbeat: () => void | Promise<void>;
};

let socket: WebSocket | null = null;
let reconnectTimer: number | undefined;
let heartbeatTimer: number | undefined;
let lastMessageAt = 0;
let stopped = false;

export function appRealtimeHealthy(): boolean {
  return Boolean(socket?.readyState === WebSocket.OPEN && Date.now() - lastMessageAt < 25_000);
}

export function startAppRealtime(hooks: AppRealtimeHooks): () => void {
  stopped = false;
  const connect = () => {
    if (stopped || socket?.readyState === WebSocket.OPEN || socket?.readyState === WebSocket.CONNECTING) return;
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${scheme}://${window.location.host}/ws/app/state/`);
    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        lastMessageAt = Date.now();
        if (payload?.type === 'app.snapshot' && payload?.summary) {
          hooks.onSnapshot(
            payload.summary,
            payload.wallet_snapshot || {},
            payload.reconciliation || {},
            String(payload.revision || ''),
          );
        }
      } catch (error) {
        console.warn('Invalid app realtime payload', error);
      }
    };
    socket.onclose = () => {
      socket = null;
      if (!stopped) reconnectTimer = window.setTimeout(connect, 3_000);
    };
    socket.onerror = () => socket?.close();
  };

  connect();
  heartbeatTimer = window.setInterval(() => {
    if (!appRealtimeHealthy()) void hooks.onFallbackHeartbeat();
    if (!socket || socket.readyState === WebSocket.CLOSED) connect();
  }, 15_000);

  return () => {
    stopped = true;
    if (reconnectTimer) window.clearTimeout(reconnectTimer);
    if (heartbeatTimer) window.clearInterval(heartbeatTimer);
    socket?.close();
    socket = null;
  };
}
