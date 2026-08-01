type WalletRealtimeHooks = {
  onSnapshot: (snapshot: any, reconciliation: any) => void;
  onFallbackHeartbeat: () => void | Promise<void>;
};

let socket: WebSocket | null = null;
let reconnectTimer: number | undefined;
let heartbeatTimer: number | undefined;
let lastMessageAt = 0;
let stopped = false;

export function walletRealtimeHealthy(): boolean {
  return Boolean(socket?.readyState === WebSocket.OPEN && Date.now() - lastMessageAt < 25_000);
}

export function startWalletRealtime(hooks: WalletRealtimeHooks): () => void {
  stopped = false;
  const connect = () => {
    if (stopped || socket?.readyState === WebSocket.OPEN || socket?.readyState === WebSocket.CONNECTING) return;
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${scheme}://${window.location.host}/ws/wallet/state/`);
    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        lastMessageAt = Date.now();
        if (payload?.snapshot) hooks.onSnapshot(payload.snapshot, payload.reconciliation || {});
      } catch (error) {
        console.warn('Invalid wallet realtime payload', error);
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
    if (!walletRealtimeHealthy()) void hooks.onFallbackHeartbeat();
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
