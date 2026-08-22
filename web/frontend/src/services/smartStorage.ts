/**
 * SmartStorageService — the browser half of the hybrid database.
 *
 * This is the local tier, and it is the whole point of the architecture:
 * AllezORM runs a real SQLite database in the tab (persisted to IndexedDB),
 * so the app queries locally and the server stays a thin, stateless S3 proxy.
 *
 *     AllezORM / IndexedDB          <- real SQL, survives reload & offline
 *          |  REST over API Gateway
 *          v
 *     Lambda (stateless)
 *          |
 *          v
 *     S3  database/tables/<table>/<id>.json
 *
 * Ported from C4rr13rX's smart-storage.service.ts, with the same three-level
 * cascade: local DB -> remote -> last-known-good.
 *
 * Why local-first is also the cheapest option
 * -------------------------------------------
 * Every read answered from IndexedDB is an API Gateway request, a Lambda
 * invocation, and an S3 GET that never happen. For a dashboard that polls,
 * that is the difference between constant billing and near-zero: the server is
 * touched on first load, on writes, and when the change feed says something
 * actually moved.
 */

import { AllezORM } from 'allez-orm';

export type Id = number | string;
export type Row = Record<string, any>;
export type Query = Record<string, string | number | boolean | null | undefined>;

/** Tables mirrored into the browser database. */
export interface TableSpec {
  /** REST resource name, which is also the S3 table name. */
  resource: string;
  /** CREATE TABLE run against the local SQLite database. */
  createSQL: string;
}

const DB_NAME = 'coolcrypto-hybrid.db';

/** How long a locally cached table is trusted before revalidating. */
const STALE_AFTER_MS = 30_000;

function toQueryString(q?: Query): string {
  if (!q) return '';
  const parts: string[] = [];
  for (const [k, v] of Object.entries(q)) {
    if (v === undefined || v === null) continue;
    parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  }
  return parts.length ? `?${parts.join('&')}` : '';
}

/** Reject a promise that outlives `ms` so a hung request cannot wedge the UI. */
function withTimeout<T>(p: Promise<T>, ms = 6000): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`timeout after ${ms}ms`)), ms);
    p.then(
      (v) => { clearTimeout(t); resolve(v); },
      (e) => { clearTimeout(t); reject(e); },
    );
  });
}

/** Normalise the shapes the API may return into a plain array. */
function normalizeArray<T = any>(result: unknown): T[] {
  if (Array.isArray(result)) return result as T[];
  if (result && typeof result === 'object') {
    const r = result as any;
    if (Array.isArray(r.items)) return r.items as T[];
    if (Array.isArray(r.data)) return r.data as T[];
    return [r as T];
  }
  return [];
}

export class SmartStorage {
  private orm: AllezORM | null = null;
  private readonly specs = new Map<string, TableSpec>();
  private readonly lastSync = new Map<string, number>();
  private readonly changeSeq = new Map<string, number>();

  /** Queued mutations made while offline, replayed on reconnect. */
  private outbox: Array<{
    resource: string; op: 'insert' | 'update' | 'delete';
    id?: Id; body?: Row; queuedAt: number;
  }> = [];

  constructor(
    private readonly apiBase: string = '/api/hybrid',
    private readonly authToken: () => string | null = () => null,
  ) {}

  // -- lifecycle -------------------------------------------------------
  async init(specs: TableSpec[]): Promise<void> {
    for (const s of specs) this.specs.set(s.resource, s);
    this.orm = await AllezORM.init({
      dbName: DB_NAME,
      schemas: specs.map((s) => ({ table: s.resource, createSQL: s.createSQL })),
    });
    this.loadOutbox();
    // A tab that comes back online should flush before the user notices.
    if (typeof window !== 'undefined') {
      window.addEventListener('online', () => { void this.flushOutbox(); });
    }
    await this.flushOutbox();
  }

  private db(): AllezORM {
    if (!this.orm) throw new Error('SmartStorage.init() has not been awaited');
    return this.orm;
  }

  // -- remote ----------------------------------------------------------
  private async call<T>(method: string, path: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const token = this.authToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(`${this.apiBase}${path}`, {
      method: method.toUpperCase(),
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: 'include',
    });
    if (!res.ok) {
      const err: any = new Error(`${method} ${path} -> ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
  }

  // -- reads -----------------------------------------------------------
  /**
   * List a table, local-first.
   *
   * 1. Serve from SQLite when the snapshot is fresh, or when the server's
   *    change sequence matches what we already have.
   * 2. Otherwise fetch, write through to SQLite, and return.
   * 3. If the network fails, return whatever SQLite holds rather than
   *    throwing -- a stale dashboard beats a blank one.
   */
  async list<T extends Row = Row>(resource: string, query?: Query): Promise<T[]> {
    const fresh = Date.now() - (this.lastSync.get(resource) ?? 0) < STALE_AFTER_MS;
    if (fresh) {
      const local = await this.localRows<T>(resource);
      if (local.length) return local;
    }

    try {
      // The change feed is one small GET; it decides whether the (much larger)
      // table fetch is needed at all.
      const remoteSeq = await this.remoteChangeSeq(resource);
      if (remoteSeq !== null && remoteSeq === this.changeSeq.get(resource)) {
        const local = await this.localRows<T>(resource);
        this.lastSync.set(resource, Date.now());
        if (local.length) return local;
      }

      const raw = await withTimeout(
        this.call<any>('get', `/${resource}${toQueryString(query)}`),
        8000,
      );
      const rows = normalizeArray<T>(raw);
      await this.replaceLocal(resource, rows);
      this.lastSync.set(resource, Date.now());
      if (remoteSeq !== null) this.changeSeq.set(resource, remoteSeq);
      return rows;
    } catch (err) {
      console.warn('[SmartStorage] list remote failed; serving local', resource, err);
      return this.localRows<T>(resource);
    }
  }

  async getById<T extends Row = Row>(resource: string, id: Id): Promise<T | undefined> {
    const local = await this.db().query<T>(
      `SELECT * FROM ${this.ident(resource)} WHERE id = ? LIMIT 1`, [id],
    );
    if (local.length) return local[0];
    try {
      const row = await this.call<T>('get', `/${resource}/${id}`);
      if (row) await this.upsertLocal(resource, row);
      return row;
    } catch {
      return undefined;
    }
  }

  /** Arbitrary SQL against the local mirror -- joins, aggregates, LIKE. */
  async query<T extends Row = Row>(sql: string, params: any[] = []): Promise<T[]> {
    return this.db().query<T>(sql, params);
  }

  // -- writes ----------------------------------------------------------
  /**
   * Writes go to the server first, then to SQLite.
   *
   * Server-first matters: S3 allocates the id, and a local row invented with a
   * guessed id would collide the moment another client inserts. When the
   * network is down the mutation is queued instead, and applied locally with a
   * temporary negative id so the UI stays responsive.
   */
  async create<T extends Row = Row>(resource: string, body: Row): Promise<T> {
    try {
      const created = await withTimeout(
        this.call<T>('post', `/${resource}`, body), 8000,
      );
      await this.upsertLocal(resource, created);
      this.changeSeq.delete(resource);
      return created;
    } catch (err) {
      console.warn('[SmartStorage] create queued offline', resource, err);
      const optimistic = { ...body, id: -Date.now() } as unknown as T;
      await this.upsertLocal(resource, optimistic);
      this.queue({ resource, op: 'insert', body, queuedAt: Date.now() });
      return optimistic;
    }
  }

  async update<T extends Row = Row>(resource: string, id: Id, body: Row): Promise<T> {
    try {
      const updated = await withTimeout(
        this.call<T>('put', `/${resource}/${id}`, body), 8000,
      );
      await this.upsertLocal(resource, updated);
      this.changeSeq.delete(resource);
      return updated;
    } catch (err) {
      console.warn('[SmartStorage] update queued offline', resource, err);
      const merged = { ...body, id } as unknown as T;
      await this.upsertLocal(resource, merged);
      this.queue({ resource, op: 'update', id, body, queuedAt: Date.now() });
      return merged;
    }
  }

  async remove(resource: string, id: Id): Promise<void> {
    try {
      await withTimeout(this.call<void>('delete', `/${resource}/${id}`), 8000);
      this.changeSeq.delete(resource);
    } catch (err) {
      console.warn('[SmartStorage] delete queued offline', resource, err);
      this.queue({ resource, op: 'delete', id, queuedAt: Date.now() });
    }
    await this.db().exec(
      `DELETE FROM ${this.ident(resource)} WHERE id = ?`, [id],
    );
  }

  // -- offline outbox --------------------------------------------------
  private queue(entry: {
    resource: string; op: 'insert' | 'update' | 'delete';
    id?: Id; body?: Row; queuedAt: number;
  }): void {
    this.outbox.push(entry);
    this.saveOutbox();
  }

  /**
   * Replay queued mutations in order.
   *
   * Stops at the first failure rather than skipping past it: the entries are
   * ordered, and applying a later update before an earlier insert would write
   * the wrong final state.
   */
  async flushOutbox(): Promise<number> {
    let sent = 0;
    while (this.outbox.length) {
      const entry = this.outbox[0];
      try {
        if (entry.op === 'insert') {
          const created = await this.call<Row>('post', `/${entry.resource}`, entry.body);
          await this.upsertLocal(entry.resource, created);
        } else if (entry.op === 'update') {
          await this.call('put', `/${entry.resource}/${entry.id}`, entry.body);
        } else {
          await this.call('delete', `/${entry.resource}/${entry.id}`);
        }
        this.outbox.shift();
        sent += 1;
      } catch {
        break; // still offline (or server rejecting) -- try again later
      }
    }
    if (sent) {
      this.saveOutbox();
      this.lastSync.clear();
    }
    return sent;
  }

  get pendingWrites(): number {
    return this.outbox.length;
  }

  private saveOutbox(): void {
    try {
      localStorage.setItem('ccu.hybrid.outbox', JSON.stringify(this.outbox));
    } catch { /* quota or private mode: the queue is best-effort */ }
  }

  private loadOutbox(): void {
    try {
      const raw = localStorage.getItem('ccu.hybrid.outbox');
      this.outbox = raw ? JSON.parse(raw) : [];
    } catch { this.outbox = []; }
  }

  // -- local helpers ---------------------------------------------------
  private ident(resource: string): string {
    // AllezORM's safeIdent equivalent: refuse anything that is not a plain
    // identifier so a resource name can never become SQL injection.
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(resource)) {
      throw new Error(`unsafe table name: ${resource}`);
    }
    return `"${resource}"`;
  }

  private async localRows<T extends Row = Row>(resource: string): Promise<T[]> {
    try {
      return await this.db().query<T>(`SELECT * FROM ${this.ident(resource)}`);
    } catch {
      return [];
    }
  }

  private async replaceLocal(resource: string, rows: Row[]): Promise<void> {
    const table = this.ident(resource);
    await this.db().exec(`DELETE FROM ${table}`);
    for (const row of rows) await this.upsertLocal(resource, row);
  }

  private async upsertLocal(resource: string, row: Row): Promise<void> {
    if (!row) return;
    const spec = this.specs.get(resource);
    if (!spec) return;
    try {
      await this.db().table(resource).upsert(row);
    } catch (err) {
      console.warn('[SmartStorage] local upsert failed', resource, err);
    }
  }

  private async remoteChangeSeq(resource: string): Promise<number | null> {
    try {
      const res = await withTimeout(
        this.call<{ seq: number }>('get', `/${resource}/_change`), 3000,
      );
      return typeof res?.seq === 'number' ? res.seq : null;
    } catch {
      return null;
    }
  }
}

export const smartStorage = new SmartStorage(
  '/api/hybrid',
  () => localStorage.getItem('ccu.auth.token'),
);
