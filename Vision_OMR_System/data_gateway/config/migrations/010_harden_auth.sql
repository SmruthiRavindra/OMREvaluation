-- Migration: 010_harden_auth.sql
-- Description: Store SHA-256 session token hashes instead of raw tokens, and re-seed default admin/faculty users with async bcrypt hashes & random passwords.

BEGIN;

-- 1. Rename token to token_hash and widen column for user_sessions
ALTER TABLE user_sessions RENAME COLUMN token TO token_hash;

-- 2. Clear out any legacy active session records (force re-login)
TRUNCATE TABLE user_sessions;

-- 3. Remove old PBKDF2 seed accounts if present
DELETE FROM users WHERE username IN ('admin', 'faculty');

DO $$
DECLARE
    admin_pass TEXT := encode(gen_random_bytes(16), 'hex');
    faculty_pass TEXT := encode(gen_random_bytes(16), 'hex');
BEGIN
    RAISE NOTICE '=======================================================';
    RAISE NOTICE 'ATTENTION OPERATOR - PILOT INITIALIZATION CREDENTIALS:';
    RAISE NOTICE 'Admin Username: admin  | Password: %', admin_pass;
    RAISE NOTICE 'Faculty Username: faculty | Password: %', faculty_pass;
    RAISE NOTICE 'PLEASE RECORD THESE PASSWORDS IMMEDIATELY!';
    RAISE NOTICE '=======================================================';

    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    INSERT INTO users (username, password_hash, salt, role)
    VALUES ('admin', crypt(admin_pass, gen_salt('bf', 12)), '', 'admin');

    INSERT INTO users (username, password_hash, salt, role)
    VALUES ('faculty', crypt(faculty_pass, gen_salt('bf', 12)), '', 'faculty');

    -- Write one-time credentials file
    EXECUTE format('COPY (SELECT %L || E''\n'' || %L) TO %L', 
        'Admin Username: admin | Password: ' || admin_pass,
        'Faculty Username: faculty | Password: ' || faculty_pass,
        '/app/.admin_credentials_ONE_TIME.txt'
    );
END $$;

COMMIT;
