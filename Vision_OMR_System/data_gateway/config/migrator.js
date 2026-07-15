/**
 * migrator.js
 * -----------
 * Helper to run SQL database migrations on startup.
 * Each migration is run independently - a failure in one does NOT block others.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import pool from './database.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export async function runMigrations() {
  console.log('[DB Migrator] Checking database migrations...');
  const migrationsDir = path.join(__dirname, 'migrations');

  // Create migrations tracker table if not exists
  await pool.query(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      filename VARCHAR(255) PRIMARY KEY,
      applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
  `);

  // Read and sort SQL migration files
  const files = fs.readdirSync(migrationsDir)
    .filter(f => f.endsWith('.sql'))
    .sort();

  let applied = 0;
  let skipped = 0;
  let failed = 0;

  for (const file of files) {
    try {
      const { rows } = await pool.query('SELECT 1 FROM schema_migrations WHERE filename = $1', [file]);
      
      if (rows.length > 0) {
        skipped++;
        continue; // Already applied
      }

      console.log(`[DB Migrator] Applying migration: ${file}`);
      const sql = fs.readFileSync(path.join(migrationsDir, file), 'utf8');
      
      // Execute migration
      await pool.query(sql);
      
      // Mark as applied
      await pool.query('INSERT INTO schema_migrations (filename) VALUES ($1)', [file]);
      console.log(`[DB Migrator] ✅ Applied: ${file}`);
      applied++;
    } catch (err) {
      // Log per-migration error but continue — don't crash the server
      console.warn(`[DB Migrator] ⚠️ Migration ${file} failed (may already be applied): ${err.message}`);
      failed++;
    }
  }
  
  console.log(`[DB Migrator] Done. Applied: ${applied}, Skipped: ${skipped}, Failed: ${failed}`);
}
