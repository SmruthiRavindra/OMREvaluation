/**
 * database.js
 * -----------
 * PostgreSQL connection pool configuration using the `pg` library.
 *
 * All database interactions in the data gateway should import the pool
 * from this module.  The pool is created once and reused across requests.
 *
 * Environment variables (set in .env):
 *   PGHOST     – Postgres host         (default: localhost)
 *   PGPORT     – Postgres port         (default: 5432)
 *   PGDATABASE – Database name         (default: omr_db)
 *   PGUSER     – DB user
 *   PGPASSWORD – DB password
 *   DB_POOL_MAX          – Max connections in pool  (default: 10)
 *   DB_IDLE_TIMEOUT_MS   – Idle connection timeout  (default: 30 000)
 *   DB_CONN_TIMEOUT_MS   – Connection acquire timeout (default: 5 000)
 */

import pg from 'pg';
import 'dotenv/config';

const { Pool } = pg;

const pool = new Pool({
  host:              process.env.PGHOST     ?? 'localhost',
  port:              Number(process.env.PGPORT ?? 5432),
  database:          process.env.PGDATABASE ?? 'omr_db',
  user:              process.env.PGUSER,
  password:          process.env.PGPASSWORD,
  max:               Number(process.env.DB_POOL_MAX         ?? 10),
  idleTimeoutMillis: Number(process.env.DB_IDLE_TIMEOUT_MS  ?? 30_000),
  connectionTimeoutMillis: Number(process.env.DB_CONN_TIMEOUT_MS ?? 5_000),
  ssl: process.env.PGSSL === 'true'
    ? { rejectUnauthorized: false }
    : false,
});

// ── Pool event hooks ───────────────────────────────────────────────────────
pool.on('connect', () => {
  console.log('[DB] New client connected to PostgreSQL pool');
});

pool.on('error', (err) => {
  console.error('[DB] Unexpected pool error:', err.message);
  // Do not crash the server; pg Pool handles reconnection automatically
});

// ── Graceful shutdown ──────────────────────────────────────────────────────
process.on('SIGINT',  () => pool.end().then(() => process.exit(0)));
process.on('SIGTERM', () => pool.end().then(() => process.exit(0)));

// ── Helper: run a query with automatic client return ──────────────────────

/**
 * Execute a parameterised SQL query.
 *
 * @param {string}  text    – SQL string with $1, $2 … placeholders
 * @param {Array}   params  – Query parameters
 * @returns {Promise<pg.QueryResult>}
 */
export async function query(text, params = []) {
  const start = Date.now();
  const result = await pool.query(text, params);
  const ms = Date.now() - start;
  if (process.env.NODE_ENV !== 'production') {
    console.debug(`[DB] query="${text.slice(0, 80)}" rows=${result.rowCount} time=${ms}ms`);
  }
  return result;
}

/**
 * Obtain a dedicated client from the pool (for transactions).
 *
 * @returns {Promise<pg.PoolClient>}
 */
export async function getClient() {
  return pool.connect();
}

export default pool;
