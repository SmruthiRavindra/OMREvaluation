-- =============================================================================
-- 004_usn_unique_constraint.sql
-- Vision OMR System — Data Gateway
--
-- Adds a unique constraint on (session_id, usn) to the student_results table.
-- This enables UPSERT logic when submitting scores.
-- =============================================================================

BEGIN;

-- Add UNIQUE constraint to prevent duplicate rows for the same student in a session
ALTER TABLE student_results 
ADD CONSTRAINT unique_session_id_usn UNIQUE (session_id, usn);

COMMIT;
