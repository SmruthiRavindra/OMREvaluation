/**
 * migrator.js
 * -----------
 * Helper to run SQL database migrations on startup.
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

  try {
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

    for (const file of files) {
      const { rows } = await pool.query('SELECT 1 FROM schema_migrations WHERE filename = $1', [file]);
      
      if (rows.length === 0) {
        console.log(`[DB Migrator] Applying migration file: ${file}`);
        const sql = fs.readFileSync(path.join(migrationsDir, file), 'utf8');
        
        // Execute migration query block
        await pool.query(sql);
        
        // Mark migration as successfully applied
        await pool.query('INSERT INTO schema_migrations (filename) VALUES ($1)', [file]);
        console.log(`[DB Migrator] Successfully applied: ${file}`);
      }
    }
    
    console.log('[DB Migrator] All migrations are up to date.');
  } catch (err) {
    console.error('[DB Migrator] Error during migrations:', err.message);
    throw err;
  }
}
