-- =============================================================================
-- 004_usn_unique_constraint.sql
-- Vision OMR System — Data Gateway
--
-- Adds a unique constraint on (session_id, usn) to the student_results table.
-- This enables UPSERT logic when submitting scores.
-- Idempotent: safe to run multiple times.
-- =============================================================================

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'unique_session_id_usn'
  ) THEN
    ALTER TABLE student_results ADD CONSTRAINT unique_session_id_usn UNIQUE (session_id, usn);
  END IF;
END $$;
