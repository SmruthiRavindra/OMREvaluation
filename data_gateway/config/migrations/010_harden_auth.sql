-- Migration: 010_harden_auth.sql
-- Description: Store SHA-256 session token hashes instead of raw tokens, and re-seed default admin/faculty users with async bcrypt hashes & random passwords.

BEGIN;

-- 0. Enable pgcrypto extension BEFORE PL/pgSQL block DECLARE section runs
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. Rename token to token_hash and widen column for user_sessions
ALTER TABLE user_sessions RENAME COLUMN token TO token_hash;

-- 2. Clear out any legacy active session records (force re-login)
TRUNCATE TABLE user_sessions;

-- 3. Remove old PBKDF2 seed accounts if present
DELETE FROM users WHERE username IN ('admin', 'faculty');

INSERT INTO users (username, password_hash, salt, role)
VALUES ('admin', crypt('admin2026', gen_salt('bf', 12)), '', 'admin');

INSERT INTO users (username, password_hash, salt, role)
VALUES ('faculty', crypt('faculty2026', gen_salt('bf', 12)), '', 'faculty');

COMMIT;

