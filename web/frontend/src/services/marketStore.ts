/**
 * Shared market data — Parquet from S3, searched locally with AllezORM.
 *
 * Market data is shared across all accounts (the price of WETH at a given
 * second is not per-user), so it is cached aggressively and carries no
 * account filter. This is the read path that pairs with
 * `serverless/hybrid/market_store.py`.
 *
 *     S3: database/market/<table>/<YYYY-MM>.parquet   (columnar, zstd)
 *          |  fetched once per month-partition, then cached
 *          v
 *     AllezORM / IndexedDB    <- the search tier: SQL over local rows
 *
 * Why partitions and not rows
 * ---------------------------
 * There are ~3.4M rows here. One JSON object per row would mean millions of
 * requests per sync; monthly Parquet turns that into ~24 objects. Parquet is
 * columnar, so a price chart transfers `ts` and `price` without ever pulling
 * the `raw` blob column.
 *
 * Partitions are immutable once a month closes, which is what makes the
 * caching safe: only the current month is ever refetched.
 */

import { parquetReadObjects } from 'hyparquet';
import { compressors } from 'hyparquet-compressors';
import { AllezORM } from 'allez-orm';

export interface PartitionInfo {
  key: string;      // "2026-07"
  rows: number;
  min_ts: number;
  max_ts: number;
  bytes: number;
}

export interface Manifest {
  table: string;
  partitions: PartitionInfo[];
  rows: number;
  bytes: number;
  shared: boolean;
}

export interface BlobIndexEntry {
  id: string;
  ts: number;
  bytes: number;
  raw_bytes: number;
}

/** Local SQLite schema per market table, created on demand. */
const MARKET_SCHEMAS: Record<string, string> = {
  market_stream: `CREATE TABLE IF NOT EXISTS market_stream (
    id INTEGER PRIMARY KEY, ts REAL, chain TEXT, symbol TEXT,
    price REAL, volume REAL, raw TEXT)`,
  metrics: `CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY, ts REAL, stage TEXT, category TEXT,
    name TEXT, value REAL, meta TEXT)`,
  feedback_events: `CREATE TABLE IF NOT EXISTS feedback_events (
    id INTEGER PRIMARY KEY, ts REAL, source TEXT, severity TEXT,
    label TEXT, details TEXT)`,
  trade_fills: `CREATE TABLE IF NOT EXISTS trade_fills (
    id INTEGER PRIMARY KEY, ts REAL, chain TEXT, symbol TEXT)`,
  trading_ops: `CREATE TABLE IF NOT EXISTS trading_ops (
    id INTEGER PRIMARY KEY, ts REAL, wallet TEXT, chain TEXT,
    symbol TEXT, action TEXT, status TEXT, details TEXT)`,
  prices: `CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY, chain TEXT, token TEXT, usd REAL,
    source TEXT, ts REAL)`,
  advisories: `CREATE TABLE IF NOT EXISTS advisories (
    id INTEGER PRIMARY KEY, ts REAL, scope TEXT, topic TEXT,
    severity TEXT, message TEXT, recommendation TEXT, meta TEXT)`,
  organism_snapshots_index: `CREATE TABLE IF NOT EXISTS organism_snapshots_index (
    id TEXT PRIMARY KEY, ts REAL, bytes INTEGER, raw_bytes INTEGER)`,
};

/** Tracks which partitions are already loaded, so a reload is not a refetch. */
const LOADED_KEY = 'ccu.market.loaded';

export class MarketStore {
  private orm: AllezORM | null = null;
  private loaded = new Set<string>();

  constructor(private readonly base: string = '/api/market') {}

  async init(): Promise<void> {
    this.orm = await AllezORM.init({
      dbName: 'coolcrypto-market.db',
      schemas: Object.entries(MARKET_SCHEMAS).map(([table, createSQL]) => ({
        table, createSQL,
      })),
    });
    try {
      const raw = localStorage.getItem(LOADED_KEY);
      if (raw) this.loaded = new Set(JSON.parse(raw));
    } catch { /* first run */ }
  }

  private db(): AllezORM {
    if (!this.orm) throw new Error('MarketStore.init() has not been awaited');
    return this.orm;
  }

  private rememberLoaded(tag: string): void {
    this.loaded.add(tag);
    try {
      localStorage.setItem(LOADED_KEY, JSON.stringify([...this.loaded]));
    } catch { /* quota: re-fetching is only a performance loss */ }
  }

  // -- manifest --------------------------------------------------------
  async manifest(table: string): Promise<Manifest | null> {
    try {
      const res = await fetch(`${this.base}/${table}/manifest`, {
        credentials: 'include',
      });
      return res.ok ? ((await res.json()) as Manifest) : null;
    } catch {
      return null;
    }
  }

  // -- partition loading -----------------------------------------------
  /**
   * Ensure the partitions covering [fromTs, toTs] are in the local database.
   *
   * Only fetches what the range needs and what is not already cached, so
   * "last 7 days" costs one partition rather than the whole table.
   */
  async ensureRange(table: string, fromTs: number, toTs: number): Promise<number> {
    const manifest = await this.manifest(table);
    if (!manifest) return 0;

    const currentMonth = new Date().toISOString().slice(0, 7);
    let fetched = 0;

    for (const part of manifest.partitions) {
      if (part.max_ts < fromTs || part.min_ts > toTs) continue;
      const tag = `${table}:${part.key}`;
      // A closed month is immutable, so a cached copy stays valid forever.
      // The current month still receives writes and must be refetched.
      if (this.loaded.has(tag) && part.key !== currentMonth) continue;
      await this.loadPartition(table, part.key);
      this.rememberLoaded(tag);
      fetched += 1;
    }
    return fetched;
  }

  private async loadPartition(table: string, month: string): Promise<void> {
    const res = await fetch(`${this.base}/${table}/partition/${month}`, {
      credentials: 'include',
    });
    if (!res.ok) throw new Error(`partition ${table}/${month} -> ${res.status}`);

    const buffer = await res.arrayBuffer();
    // hyparquet needs the compression codecs supplied explicitly; the
    // partitions are zstd, which is not built into the base reader.
    const rows = await parquetReadObjects({ file: buffer, compressors });

    const db = this.db();
    // One transaction for the whole partition: committing per row turns a
    // 200k-row month into 200k IndexedDB writes.
    await db.exec('BEGIN');
    try {
      for (const row of rows) {
        await db.table(table).upsert(row as Record<string, any>);
      }
      await db.exec('COMMIT');
    } catch (err) {
      await db.exec('ROLLBACK');
      throw err;
    }
  }

  // -- local search ----------------------------------------------------
  /** Arbitrary SQL against the locally mirrored market data. */
  async query<T = Record<string, any>>(sql: string, params: any[] = []): Promise<T[]> {
    return this.db().query<T>(sql, params);
  }

  /** Price series for one symbol, loading only the partitions it needs. */
  async priceSeries(symbol: string, fromTs: number, toTs: number) {
    await this.ensureRange('market_stream', fromTs, toTs);
    return this.query<{ ts: number; price: number }>(
      `SELECT ts, price FROM market_stream
        WHERE symbol = ? AND ts BETWEEN ? AND ?
        ORDER BY ts`,
      [symbol, fromTs, toTs],
    );
  }

  /** Named metric over a window. */
  async metricSeries(name: string, fromTs: number, toTs: number) {
    await this.ensureRange('metrics', fromTs, toTs);
    return this.query<{ ts: number; value: number; stage: string }>(
      `SELECT ts, value, stage FROM metrics
        WHERE name = ? AND ts BETWEEN ? AND ?
        ORDER BY ts`,
      [name, fromTs, toTs],
    );
  }

  // -- organism snapshots (index local, payloads remote) ---------------
  /**
   * Mirror the snapshot *index* only.
   *
   * The payloads total ~22 GB raw. Only their timestamps and sizes live
   * locally; a body is fetched individually when the user opens one, which is
   * the only reason this table is usable in a browser at all.
   */
  async loadSnapshotIndex(): Promise<number> {
    const tag = 'organism_snapshots:index';
    if (this.loaded.has(tag)) {
      const [{ n }] = await this.query<{ n: number }>(
        'SELECT COUNT(*) AS n FROM organism_snapshots_index');
      return n;
    }
    const res = await fetch(`${this.base}/organism_snapshots/index`, {
      credentials: 'include',
    });
    if (!res.ok) return 0;

    const rows = await parquetReadObjects({
      file: await res.arrayBuffer(), compressors,
    });
    const db = this.db();
    await db.exec('BEGIN');
    try {
      for (const row of rows) {
        await db.table('organism_snapshots_index').upsert(row as any);
      }
      await db.exec('COMMIT');
    } catch (err) {
      await db.exec('ROLLBACK');
      throw err;
    }
    this.rememberLoaded(tag);
    return rows.length;
  }

  /** Fetch one snapshot payload on demand. */
  async snapshot(id: string): Promise<any | null> {
    const res = await fetch(`${this.base}/organism_snapshots/blob/${id}`, {
      credentials: 'include',
    });
    return res.ok ? res.json() : null;
  }

  /** How much of the local mirror is populated. */
  async stats(): Promise<Record<string, number>> {
    const out: Record<string, number> = {};
    for (const table of Object.keys(MARKET_SCHEMAS)) {
      try {
        const [{ n }] = await this.query<{ n: number }>(
          `SELECT COUNT(*) AS n FROM "${table}"`);
        out[table] = n;
      } catch {
        out[table] = 0;
      }
    }
    return out;
  }
}

export const marketStore = new MarketStore();
